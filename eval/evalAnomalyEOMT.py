# ==============================================================================
# SCRIPT DI VALUTAZIONE OOD (Out-Of-Distribution) PER EoMT / Mask2Former
# ==============================================================================
# Questo script valuta la capacità di un modello semantico (EoMT) di rilevare 
# anomalie e ostacoli sconosciuti su strada. Utilizza 4 metodi di scoring OOD:
# 1. MSP (Maximum Softmax Probability)
# 2. MaxLogit (Maximum Logit)
# 3. MaxEntropy (Maximum Entropy)
# 4. RbA (Rejection by Anticipation)
# ==============================================================================

import os
import glob
import torch
import random
import yaml
import importlib
from PIL import Image
import numpy as np
import gc

# Import dinamico del framework EoMT
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../eomt')))

from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
import torch.nn.functional as F

# --- Fissaggio del Seme (Seed) per la Riproducibilità ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# --- Costanti Globali ---
NUM_CHANNELS = 3

device = 0



def compute_anomaly_score(semantic_scores, method='msp'):
    """
    Calcola lo score di anomalia per ogni pixel. Uno score alto indica una maggiore probabilità
    che il pixel sia un'anomalia (Out-of-Distribution).

    INPUT:
    - semantic_scores: Tensor [C, H, W] — gli scores per-pixel di MaskFormer, ottenuti da
      to_per_pixel_logits_semantic (= sum_q sigmoid(mask_q) * softmax(cls_q)).
      NON sono probabilità vere (non sommano a 1), ma sono gli scores ufficiali su cui
      il modello fa argmax per la segmentazione semantica.

    METODI:
    - MSP: Applica Softmax agli scores per ottenere probabilità vere, poi 1 - max(P).
    - MaxLogit: -max(score). Usa gli scores MaskFormer direttamente (senza Softmax).
    - MaxEntropy: Applica Softmax, poi entropia di Shannon -sum(P * log(P)).
    - RbA (Nayal et al. 2022): -sum(tanh(score)). Usa gli scores MaskFormer direttamente.
    """
    if method == 'maxlogit':
        scores_np = semantic_scores.cpu().numpy()  # [C, H, W]
        return -np.max(scores_np, axis=0)

    elif method == 'msp':
        # Normalizziamo con Softmax per ottenere probabilità vere che sommano a 1
        probs = F.softmax(semantic_scores.unsqueeze(0).float(), dim=1).squeeze(0)
        probs_np = probs.cpu().numpy()
        return 1.0 - np.max(probs_np, axis=0)

    elif method == 'maxentropy':
        # Normalizziamo con Softmax per ottenere probabilità vere
        probs = F.softmax(semantic_scores.unsqueeze(0).float(), dim=1).squeeze(0)
        probs_np = probs.cpu().numpy()
        probs_np = np.clip(probs_np, 1e-9, 1.0)
        return -np.sum(probs_np * np.log(probs_np), axis=0)

    elif method == 'rba':
        rba = -torch.tanh(semantic_scores.float()).sum(dim=0)
        return rba.cpu().numpy()

    else:
        raise ValueError(f"Metodo '{method}' non riconosciuto.")


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_dir", default="/content/drive/MyDrive/project/Anomaly_Validation_Datasets/Validation_Dataset")
    parser.add_argument('--model', type=str, default='cityscapes', 
                        choices=['cityscapes', 'coco', 'coco_finetuned'],
                        help='Modello da utilizzare per la valutazione')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    DATASETS = {
        "FS_LostFound_full": "png",
        # "RoadAnomaly":       "jpg",
        # "RoadAnomaly21":     "png",
        # "RoadObsticle21":    "webp",
        # "fs_static":         "jpg"
    }

    methods_to_evaluate = ['msp', 'maxlogit', 'maxentropy', 'rba']

    MODEL_CONFIGS = {
        "cityscapes": {
            "config_path": "../eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
            "state_dict_path": "/content/drive/MyDrive/project/eomt_cityscapes.bin",
            "num_classes": 19,
            "img_size": (1024, 1024),
            "model_type": "semantic",
        },
        "coco": {
            "config_path": "../eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml",
            "state_dict_path": "/content/drive/MyDrive/project/eomt_coco.bin",
            "num_classes": 133,
            "img_size": (640, 640),
            "model_type": "panoptic",
        },
        "coco_finetuned": {
            "config_path": "../eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
            "state_dict_path": "/content/drive/MyDrive/project/eomt_coco_finetuned.bin",
            "num_classes": 19,
            "img_size": (640, 640),
            "model_type": "semantic",
        },
    }

    model_config = MODEL_CONFIGS[args.model]
    config_path = model_config["config_path"]
    state_dict_path = model_config["state_dict_path"]
    num_classes = model_config["num_classes"]
    img_size = model_config["img_size"]
    
    results_file = f'results_eomt_{args.model}.txt'
    if not os.path.exists(results_file):
        open(results_file, 'w').close()

    # ==============================================================================
    # 1. INIZIALIZZAZIONE DEL MODELLO EOMT DA CONFIGURAZIONE
    # ==============================================================================
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Inizializza l'Encoder (Backbone ViT/Dinov2)
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    # Inizializza il Network (L'architettura EoMT vera e propria basata su Mask2Former)
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(masked_attn_enabled=False, num_classes=num_classes, encoder=encoder, **network_kwargs)

    # Inizializza il Lightning Module (Wrapper per PyTorch Lightning)
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    
    if model_config["model_type"] == "panoptic":
        # Passiamo le 53 stuff classes usate in COCO Panoptic (ID da 80 a 132 compresi)
        model_kwargs["stuff_classes"] = list(range(80, 133))

    model = lit_cls(img_size=img_size, num_classes=num_classes, network=network, **model_kwargs)
    
    # Carica i pesi pre-addestrati
    print(f"Caricamento pesi da: {state_dict_path}")
    state_dict = torch.load(
        state_dict_path, map_location=f"cuda:{device}", weights_only=True
    )
    model.load_state_dict(state_dict, strict=False)

    # Se usiamo la GPU, avvolgiamo il modello in DataParallel
    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    model.eval()

    # Riferimento al modello base (senza wrapper DataParallel) indispensabile per 
    # poter chiamare i metodi custom della classe, come window_imgs_semantic()
    base_model = model.module if hasattr(model, 'module') else model

    print("Modello EoMT caricato con successo.\n")

    # ==============================================================================
    # 2. CICLO DI VALUTAZIONE SUI DATASET (E INFERENZA IMMAGINI)
    # ==============================================================================
    with open(results_file, 'a') as file:

        for dataset_name, ext in DATASETS.items():
            print(f"==================================================")
            print(f" Inizio valutazione Dataset: {dataset_name}")
            print(f"==================================================")
            file.write(f"\n\n=== Dataset: {dataset_name} ===\n")

            search_pattern = os.path.join(args.base_dir, dataset_name, "images", f"*.{ext}")
            image_paths = glob.glob(search_pattern)

            if len(image_paths) == 0:
                print(f"  Nessuna immagine trovata in {search_pattern}. Skipping.\n")
                continue

            # Accumuliamo i pixel estratti (OOD e In-Distribution) direttamente per immagine.
            # Questo approccio salva gli scores in memoria per poi processarli tutti alla fine.
            ood_scores_lists = {method: [] for method in methods_to_evaluate}
            ind_scores_lists = {method: [] for method in methods_to_evaluate}

            for img_idx, path in enumerate(image_paths):
                print(f"\r  [{img_idx+1}/{len(image_paths)}] Elaborando {os.path.basename(path)}...", end="", flush=True)
                
                # --- A. Caricamento della Ground Truth (Maschera) ---
                pathGT = path.replace("images", "labels_masks")
                base_path_gt, _ = os.path.splitext(pathGT)
                pathGT = base_path_gt + ".png"

                if not os.path.exists(pathGT):
                    continue

                # --- B. Caricamento e Preparazione Immagine Originale ---
                # Il windowing (crop) di EoMT si aspetta in input un tensore uint8 su CPU [C, H, W]
                # Questo perché la logica interna divide le immagini usando librerie come PIL
                img_pil = Image.open(path).convert('RGB')
                orig_h, orig_w = img_pil.height, img_pil.width
                img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)

                # Assicuriamoci che la maschera abbia la stessa identica risoluzione dell'immagine
                mask_pil = Image.open(pathGT)
                if mask_pil.size != (orig_w, orig_h):
                    mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
                ood_gts = np.array(mask_pil)

                # --- C. Mappatura Etichette dei Pixel ---
                # Uniformiamo le classi dei diversi dataset al seguente standard:
                # 1   = Anomalia / Ostacolo (Out-of-Distribution)
                # 0   = Strada / Normale (In-Distribution)
                # 255 = Ignora (Cielo, Auto, o pixel ambigui)
                if "RoadAnomaly" in pathGT:
                    ood_gts = np.where((ood_gts == 2), 1, ood_gts)
                if "LostAndFound" in pathGT:
                    ood_gts = np.where((ood_gts == 0), 255, ood_gts)
                    ood_gts = np.where((ood_gts == 1), 0, ood_gts)
                    ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
                if "Streethazard" in pathGT:
                    ood_gts = np.where((ood_gts == 14), 255, ood_gts)
                    ood_gts = np.where((ood_gts < 20), 0, ood_gts)
                    ood_gts = np.where((ood_gts == 255), 1, ood_gts)

                # Se in questa immagine non c'è nemmeno un pixel di anomalia, saltiamo l'immagine.
                if 1 not in np.unique(ood_gts):
                    continue

                with torch.no_grad():
                    # --- D. WINDOWING E INFERENZA ---
                    # Dato che l'immagine può essere molto grande (es. 1024x2048), la dividiamo
                    # in finestre (crop) più piccole (1024x1024) per processarla pezzo per pezzo
                    # senza esaurire la memoria della GPU.
                    img_sizes = [(orig_h, orig_w)]
                    imgs = [img_tensor]

                    # 1. Suddivisione in Crop (Su CPU)
                    crops, origins = base_model.window_imgs_semantic(imgs)

                    # 2. Spostamento su GPU e conversione in float
                    if not args.cpu:
                        crops = crops.float().cuda()
                    else:
                        crops = crops.float()

                    # 3. Inferenza di rete: ottiene predizioni per le maschere e le classi per ogni blocco
                    # mask_logits_per_block: output delle query per definire le aree delle maschere
                    # class_logits_per_block: output per la classificazione semantica delle query
                    mask_logits_per_block, class_logits_per_block = model(crops)

                    # 4. Estrazione dell'ultimo blocco del transformer (il livello più raffinato)
                    mask_logits = mask_logits_per_block[-1]
                    class_logits = class_logits_per_block[-1]

                    # 5. Ri-scalare le predizioni delle maschere alla dimensione del crop
                    mask_logits = F.interpolate(
                        mask_logits,
                        size=img_size,
                        mode="bilinear",
                        align_corners=False
                    )

                    # --- E. ESTRAZIONE DEGLI SCORES SEMANTICI PER-PIXEL ---
                    # to_per_pixel_logits_semantic calcola: sum_q sigmoid(mask_q) * softmax(cls_q)
                    # Il risultato sono gli "scores semantici" di MaskFormer [B, C, H, W].
                    # NON sono probabilità vere (non sommano a 1 sulle classi), ma sono gli
                    # scores ufficiali su cui il modello fa argmax per la segmentazione.
                    # Tutti e 4 i metodi OOD (MSP, MaxLogit, MaxEntropy, RbA) usano questi scores.
                    crop_scores = base_model.to_per_pixel_logits_semantic(mask_logits, class_logits)
                    # Riassembla i crop per ricostruire l'immagine intera originaria (orig_h, orig_w)
                    scores_list = base_model.revert_window_logits_semantic(crop_scores, origins, img_sizes)
                    semantic_scores = scores_list[0]  # [C, orig_h, orig_w]
                    del crop_scores, scores_list

                # Prepariamo le maschere booleane per recuperare velocemente i pixel validi
                ood_mask = (ood_gts == 1)
                ind_mask = (ood_gts == 0)

                # --- F. CALCOLO E SALVATAGGIO DEGLI SCORES ANOMALIA ---
                for method in methods_to_evaluate:
                    # Tutti i metodi usano lo stesso tensore semantic_scores.
                    # MSP e MaxEntropy applicano internamente il Softmax per normalizzare.
                    # MaxLogit e RbA usano gli scores MaskFormer direttamente.
                    anomaly_result = compute_anomaly_score(semantic_scores, method=method)

                    ood_scores_lists[method].append(anomaly_result[ood_mask].astype(np.float32))
                    ind_scores_lists[method].append(anomaly_result[ind_mask].astype(np.float32))

                # --- G. PULIZIA MEMORIA E GARBAGE COLLECTION ---
                del mask_logits_per_block, class_logits_per_block, mask_logits, class_logits
                del semantic_scores
                gc.collect()

            print("", flush=True)  # Vai a capo alla fine dell'elaborazione delle immagini del dataset
            
            # Verifica di sicurezza: controlliamo di aver raccolto dati validi in almeno un'immagine
            has_data = any(len(arr) > 0 for arr in ood_scores_lists[methods_to_evaluate[0]])
            if not has_data:
                print(f"  Nessuna anomalia valida trovata per {dataset_name}. Skipping dataset.\n")
                continue

            # ==============================================================================
            # 3. CALCOLO DELLE METRICHE FINALI (AUPRC e FPR@TPR95)
            # ==============================================================================
            for method in methods_to_evaluate:
                # Concateniamo tutti gli scores accumulati dalle immagini di tutto il dataset
                ood_out = np.concatenate(ood_scores_lists[method])
                ind_out = np.concatenate(ind_scores_lists[method])

                # Assegniamo le etichette binarie per il calcolo matematico
                # 1 = Positivo (Outlier/Anomalo), 0 = Negativo (Inlier/Normale)
                ood_label = np.ones(len(ood_out), dtype=np.uint8)
                ind_label = np.zeros(len(ind_out), dtype=np.uint8)

                # Uniamo tutto in un singolo grande array (Scores e Labels)
                val_out   = np.concatenate((ind_out, ood_out))
                val_label = np.concatenate((ind_label, ood_label))

                # Calcolo Area Under Precision-Recall Curve (AUPRC) tramite scikit-learn
                prc_auc = average_precision_score(val_label, val_out)
                
                # Calcolo False Positive Rate (FPR) al 95% di True Positive Rate (TPR)
                fpr     = fpr_at_95_tpr(val_out, val_label)

                # Stampa a video e scrittura su file
                print(f' -> {method.upper():10} | AUPRC: {prc_auc * 100.0:.2f}% | FPR@TPR95: {fpr * 100.0:.2f}%')
                file.write(f'Method: {method.upper():10} | AUPRC: {prc_auc * 100.0:.4f} | FPR@TPR95: {fpr * 100.0:.4f}\n')

                # Liberiamo immediatamente la memoria RAM per i risultati di questo metodo 
                # prima di procedere al successivo, garantendo sicurezza contro i Memory Leak
                del ood_out, ind_out, ood_label, ind_label, val_out, val_label
                gc.collect()

            print("\n")


if __name__ == '__main__':
    main()