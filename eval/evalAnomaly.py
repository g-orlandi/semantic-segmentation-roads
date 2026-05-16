# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn.functional as F

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def compute_anomaly_score(logits, method):
    logits_np = logits.data.cpu().numpy()  # [C, H, W]

    if method == 'maxlogit':
        anomaly_result = -np.max(logits_np, axis=0)

    elif method == 'msp':
        logits_t = logits.unsqueeze(0).float()          # [1, C, H, W]
        softmax = F.softmax(logits_t, dim=1).squeeze(0) # [C, H, W]
        softmax_np = softmax.data.cpu().numpy()
        anomaly_result = 1.0 - np.max(softmax_np, axis=0)

    elif method == 'maxentropy':
        logits_t = logits.unsqueeze(0).float()          # [1, C, H, W]
        softmax = F.softmax(logits_t, dim=1).squeeze(0) # [C, H, W]
        softmax_np = softmax.data.cpu().numpy()
        softmax_np = np.clip(softmax_np, 1e-9, 1.0)
        entropy = -np.sum(softmax_np * np.log(softmax_np), axis=0)  # [H, W]
        anomaly_result = entropy

    else:
        raise ValueError(f"Unknown method '{method}'")

    return anomaly_result


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="/content/drive/MyDrive/project/Anomaly_Validation_Datasets/Validation_Dataset",
        help="Percorso della cartella principale che contiene le sottocartelle dei 5 dataset",
    )
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()

    # Mappatura dei dataset e relative estensioni delle immagini
    DATASETS = {
        "FS_LostFound_full": "png",
        "RoadAnomaly": "jpg",
        "RoadAnomaly21": "png",
        "RoadObsticle21": "webp",
        "fs_static": "jpg"
    }

    methods_to_evaluate = ['msp', 'maxlogit', 'maxentropy']

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()

    modelpath = os.path.join(args.loadDir, args.loadModel)
    weightspath = os.path.join(args.loadDir, args.loadWeights)

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict): 
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights LOADED successfully\n")
    model.eval()

    # Apriamo il file una sola volta per aggiungere i risultati
    with open('results.txt', 'a') as file:
        
        # Iteriamo su tutti e 5 i dataset
        for dataset_name, ext in DATASETS.items():
            print(f"==================================================")
            print(f" Inizio valutazione Dataset: {dataset_name}")
            print(f"==================================================")
            file.write(f"\n\n=== Dataset: {dataset_name} ===\n")

            search_pattern = os.path.join(args.base_dir, dataset_name, "images", f"*.{ext}")
            image_paths = glob.glob(search_pattern)

            if len(image_paths) == 0:
                print(f"ATTENZIONE: Nessuna immagine trovata in {search_pattern}")
                file.write(f"Nessuna immagine trovata.\n")
                continue

            # Inizializza le liste per questo specifico dataset
            anomaly_score_lists = {method: [] for method in methods_to_evaluate}
            ood_gts_list = []

            for path in image_paths:
                # 1. Carica e prepara la Ground Truth
                pathGT = path.replace("images", "labels_masks")
                # Indipendentemente da jpg o webp, la ground truth è sempre un .png
                base_path_gt, _ = os.path.splitext(pathGT)
                pathGT = base_path_gt + ".png"

                if not os.path.exists(pathGT):
                    continue

                mask = Image.open(pathGT)
                mask = target_transform(mask)
                ood_gts = np.array(mask)

                # Mappatura etichette
                if "RoadAnomaly" in pathGT: # Copre sia RoadAnomaly che RoadAnomaly21
                    ood_gts = np.where((ood_gts == 2), 1, ood_gts)
                if "LostAndFound" in pathGT in pathGT:
                    ood_gts = np.where((ood_gts == 0), 255, ood_gts)
                    ood_gts = np.where((ood_gts == 1), 0, ood_gts)
                    ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
                if "Streethazard" in pathGT:
                    ood_gts = np.where((ood_gts == 14), 255, ood_gts)
                    ood_gts = np.where((ood_gts < 20), 0, ood_gts)
                    ood_gts = np.where((ood_gts == 255), 1, ood_gts)

                if 1 not in np.unique(ood_gts):
                    # Nessuna anomalia, saltiamo
                    continue
                
                ood_gts_list.append(ood_gts)

                # 2. Inferenza del modello
                images = input_transform(Image.open(path).convert('RGB')).unsqueeze(0).float()
                if not args.cpu:
                    images = images.cuda()

                with torch.no_grad():
                    result = model(images)  # [1, C, H, W]

                logits = result.squeeze(0)

                # 3. Calcoliamo gli score per i 3 metodi
                for method in methods_to_evaluate:
                    anomaly_result = compute_anomaly_score(logits, method)
                    anomaly_score_lists[method].append(anomaly_result)

                del result, logits, ood_gts, mask
                torch.cuda.empty_cache()

            # --- Calcolo metriche per il dataset corrente ---
            if len(ood_gts_list) == 0:
                print(f"Nessun dato valido (anomalia) trovato in {dataset_name}.")
                file.write("Nessuna anomalia rilevata nelle maschere.\n")
                continue

            ood_gts = np.array(ood_gts_list)
            ood_mask = (ood_gts == 1)
            ind_mask = (ood_gts == 0)

            for method in methods_to_evaluate:
                anomaly_scores = np.array(anomaly_score_lists[method])
                
                ood_out = anomaly_scores[ood_mask]
                ind_out = anomaly_scores[ind_mask]

                ood_label = np.ones(len(ood_out))
                ind_label = np.zeros(len(ind_out))

                val_out = np.concatenate((ind_out, ood_out))
                val_label = np.concatenate((ind_label, ood_label))

                prc_auc = average_precision_score(val_label, val_out)
                fpr = fpr_at_95_tpr(val_out, val_label)

                print(f' -> {method.upper()} | AUPRC: {prc_auc * 100.0:.2f}% | FPR@TPR95: {fpr * 100.0:.2f}%')
                file.write(f'Method: {method.upper()} | AUPRC: {prc_auc * 100.0:.4f} | FPR@TPR95: {fpr * 100.0:.4f}\n')

            print("\n")


if __name__ == '__main__':
    main()