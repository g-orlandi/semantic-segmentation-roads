# ==============================================================================
# OOD (Out-of-Distribution) EVALUATION SCRIPT FOR EoMT / Mask2Former
# ==============================================================================
# This script evaluates the ability of a semantic segmentation model (EoMT) to detect
# anomalies and unknown road obstacles. It uses four OOD scoring methods:
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
from PIL import Image
import numpy as np
import gc

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "eomt"))


from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
import torch.nn.functional as F

from utils.model_loading import build_model, load_weights
from utils.label_mapping import aggregate_coco_scores_to_cityscapes

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

NUM_CHANNELS = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_anomaly_score(semantic_scores, method='msp'):
    """
    Computes the anomaly score for each pixel. A high score indicates a higher probability
    that the pixel is anomalous, i.e. Out-of-Distribution.

    INPUT:
    - semantic_scores: Tensor [C, H, W] — per-pixel MaskFormer scores obtained from
    to_per_pixel_logits_semantic (= sum_q sigmoid(mask_q) * softmax(cls_q)).
    These are NOT true probabilities, since they do not necessarily sum to 1, but they are
    the official scores used by the model to perform the semantic segmentation argmax.

    METHODS:
    - MSP: Applies Softmax to the scores to obtain proper class probabilities, then computes
    1 - max(P).
    - MaxLogit: Computes -max(score). It uses the MaskFormer scores directly, without Softmax.
    - MaxEntropy: Applies Softmax, then computes the Shannon entropy -sum(P * log(P)).
    - RbA (Nayal et al. 2022): Computes -sum(tanh(score)). It uses the MaskFormer scores directly.
    """

    if method == 'maxlogit':
        scores_np = semantic_scores.cpu().numpy()  # [C, H, W]
        return -np.max(scores_np, axis=0)

    elif method == 'msp':
        probs = F.softmax(semantic_scores.unsqueeze(0).float(), dim=1).squeeze(0)
        probs_np = probs.cpu().numpy()
        return 1.0 - np.max(probs_np, axis=0)

    elif method == 'maxentropy':
        probs = F.softmax(semantic_scores.unsqueeze(0).float(), dim=1).squeeze(0)
        probs_np = probs.cpu().numpy()
        probs_np = np.clip(probs_np, 1e-9, 1.0)
        return -np.sum(probs_np * np.log(probs_np), axis=0)

    elif method == 'rba':
        rba = -torch.tanh(semantic_scores.float()).sum(dim=0)
        return rba.cpu().numpy()

    else:
        raise ValueError(f"Method '{method}' not recognized")


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_dir", default="/content/drive/MyDrive/project/Anomaly_Validation_Datasets/Validation_Dataset")
    parser.add_argument('--model', type=str, default='cityscapes', 
                        choices=['cityscapes', 'coco', 'coco_finetuned'],
                        help='Model to use for evaluation')
    parser.add_argument('--save_inf', action='store_true')
    args = parser.parse_args()

    DATASETS = {
        "FS_LostFound_full": "png",
        "RoadAnomaly":       "jpg",
        "RoadAnomaly21":     "png",
        "RoadObsticle21":    "webp",
        "fs_static":         "jpg"
    }

    methods_to_evaluate = ['msp', 'maxlogit', 'maxentropy', 'rba']

    MODEL_CONFIGS = {
        "cityscapes": {
            "config_path": "../eomt/configs/eomt_base_640_cs.yaml",
            "state_dict_path": "/content/drive/MyDrive/project/models_weights/eomt_cityscapes.bin",
            "num_classes": 19,
            "img_size": (1024, 1024),
        },
        "coco": {
            "config_path": "../eomt/configs/eomt_base_640_coco.yaml",
            "state_dict_path": "/content/drive/MyDrive/project/models_weights/eomt_coco.bin",
            "num_classes": 133,
            "img_size": (640, 640),
        },
        "coco_finetuned": {
            "config_path": "../eomt/configs/eomt_base_640_cs.yaml",
            "state_dict_path": "/content/drive/MyDrive/project/models_weights/eomt_coco_finetuned.bin",
            "num_classes": 19,
            "img_size": (640, 640),
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

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    is_coco = (args.model == 'coco') 
    model = build_model(config, img_size, num_classes, coco=is_coco, masked_attn_enabled=True).eval()

    print(f"Loaded weights: {state_dict_path}")
    model = load_weights(model, state_dict_path, device)

    model = torch.nn.DataParallel(model).to(device)

    model.eval()

    base_model = model.module if hasattr(model, 'module') else model

    print("EOMT model loaded.\n")

    with open(results_file, 'a') as file:

        for dataset_name, ext in DATASETS.items():
            print(f"==================================================")
            print(f" {dataset_name}")
            print(f"==================================================")
            file.write(f"\n\n=== Dataset: {dataset_name} ===\n")

            search_pattern = os.path.join(args.base_dir, dataset_name, "images", f"*.{ext}")
            image_paths = glob.glob(search_pattern)

            if len(image_paths) == 0:
                print(f"  No image found in {search_pattern}. Skipping.\n")
                continue

            ood_scores_lists = {method: [] for method in methods_to_evaluate}
            ind_scores_lists = {method: [] for method in methods_to_evaluate}

            for img_idx, path in enumerate(image_paths):
                print(f"\r  [{img_idx+1}/{len(image_paths)}] Processing {os.path.basename(path)}...", end="", flush=True)
                
                pathGT = path.replace("images", "labels_masks")
                base_path_gt, _ = os.path.splitext(pathGT)
                pathGT = base_path_gt + ".png"

                if not os.path.exists(pathGT):
                    continue

                img_pil = Image.open(path).convert('RGB')
                orig_h, orig_w = img_pil.height, img_pil.width
                img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)

                mask_pil = Image.open(pathGT)
                if mask_pil.size != (orig_w, orig_h):
                    mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
                ood_gts = np.array(mask_pil)

                if dataset_name == "RoadAnomaly":
                    raw = ood_gts.copy()
                
                    mapped = np.full_like(raw, 255)
                    mapped[raw == 0] = 0      # in-distribution / valid background
                    mapped[raw == 2] = 1      # anomaly / OOD
                
                    ood_gts = mapped

                if 1 not in np.unique(ood_gts):
                    continue

                with torch.no_grad():
                    # --- D. WINDOWING AND INFERENCE ---
                    # Since the image can be very large, e.g. 1024x2048, we split it into
                    # smaller windows/crops, e.g. 1024x1024, to process it piece by piece
                    # without exhausting GPU memory.
                    
                    img_sizes = [(orig_h, orig_w)]
                    imgs = [img_tensor]

                    # 1. Split the image into crops
                    crops, origins = base_model.window_imgs_semantic(imgs)

                    # 2. Transfer to GPU and conversion to float
                    crops = crops.float().to(device)

                    # 3. Network inference: obtains mask and class predictions for each block
                    # mask_logits_per_block: query outputs used to define the mask regions
                    # class_logits_per_block: outputs used for semantic classification of the queries
                    mask_logits_per_block, class_logits_per_block = model(crops)

                    # 4. Extraction of the last transformer block, i.e. the most refined level
                    mask_logits = mask_logits_per_block[-1]
                    class_logits = class_logits_per_block[-1]

                    # 5. Rescale the mask predictions to the crop size
                    mask_logits = F.interpolate(
                        mask_logits,
                        size=img_size,
                        mode="bilinear",
                        align_corners=False
                    )

                    # --- EXTRACTION OF PER-PIXEL SEMANTIC SCORES ---
                    # to_per_pixel_logits_semantic computes: sum_q sigmoid(mask_q) * softmax(cls_q)
                    # The result is the MaskFormer semantic score tensor [B, C, H, W].
                    # These are NOT true probabilities, since they do not necessarily sum to 1 across classes,
                    # but they are the official scores used by the model to perform the semantic segmentation argmax.
                    # All four OOD methods (MSP, MaxLogit, MaxEntropy, RbA) use these scores.
                    crop_scores = base_model.to_per_pixel_logits_semantic(mask_logits, class_logits)

                    scores_list = base_model.revert_window_logits_semantic(crop_scores, origins, img_sizes)
                    semantic_scores = scores_list[0]  # [C, orig_h, orig_w]

                    if args.model == 'coco':
                        semantic_scores = aggregate_coco_scores_to_cityscapes(semantic_scores)
                    
                    if args.save_inf:
                        save_dir = f"/content/drive/MyDrive/project/saved_logits/{args.model}/{dataset_name}"
                        os.makedirs(save_dir, exist_ok=True)
                        img_name = os.path.splitext(os.path.basename(path))[0]
                        torch.save(semantic_scores.float().cpu(), os.path.join(save_dir, f"{img_name}.pt"))
                    
                    del crop_scores, scores_list
            

                ood_mask = (ood_gts == 1)
                ind_mask = (ood_gts == 0)


                for method in methods_to_evaluate:
                    anomaly_result = compute_anomaly_score(semantic_scores, method=method)

                    ood_scores_lists[method].append(anomaly_result[ood_mask].astype(np.float32))
                    ind_scores_lists[method].append(anomaly_result[ind_mask].astype(np.float32))

                del mask_logits_per_block, class_logits_per_block, mask_logits, class_logits
                del semantic_scores
                gc.collect()

            print("", flush=True)  

            has_data = any(len(arr) > 0 for arr in ood_scores_lists[methods_to_evaluate[0]])
            if not has_data:
                print(f" No anomaly found {dataset_name}. Skipping dataset.\n")
                continue

            for method in methods_to_evaluate:
                ood_out = np.concatenate(ood_scores_lists[method])
                ind_out = np.concatenate(ind_scores_lists[method])

                ood_label = np.ones(len(ood_out), dtype=np.uint8)
                ind_label = np.zeros(len(ind_out), dtype=np.uint8)

                val_out   = np.concatenate((ind_out, ood_out))
                val_label = np.concatenate((ind_label, ood_label))

                prc_auc = average_precision_score(val_label, val_out)
                
                fpr = fpr_at_95_tpr(val_out, val_label)

                print(f' -> {method.upper():10} | AUPRC: {prc_auc * 100.0:.2f}% | FPR@TPR95: {fpr * 100.0:.2f}%')
                file.write(f'Method: {method.upper():10} | AUPRC: {prc_auc * 100.0:.4f} | FPR@TPR95: {fpr * 100.0:.4f}\n')

                del ood_out, ind_out, ood_label, ind_label, val_out, val_label
                gc.collect()

            print("\n")


if __name__ == '__main__':
    main()