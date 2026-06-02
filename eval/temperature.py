import os
import glob
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
from ood_metrics import fpr_at_95_tpr

def compute_anomaly_scaled(semantic_scores, T=1.0):
    scores_t = semantic_scores.unsqueeze(0).float()
    
    logits_scaled = scores_t / T
    probs = F.softmax(logits_scaled, dim=1).squeeze(0)
    return 1.0 - np.max(probs.numpy(), axis=0)
 
 
def main():
    base_dir = "/content/drive/MyDrive/project/Anomaly_Validation_Datasets/Validation_Dataset"
    logits_base_dir = "/content/drive/MyDrive/project/saved_logits"
    
    DATASETS = {
        "RoadAnomaly": "jpg",
        "RoadAnomaly21": "png",
        "RoadObsticle21": "webp",
        "fs_static": "jpg",
        "FS_LostFound_full": "png"
    }
    
    MODELS_TO_TEST = ['cityscapes']
    TEMPERATURES = [0.25, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0]
    
    for model_name in MODELS_TO_TEST:
        print(f"\n=======================================================")
        print(f" OTTIMIZZAZIONE TEMPERATURA PER MODELLO: {model_name.upper()}")
        print(f"=======================================================")
        
        for dataset_name, ext in DATASETS.items():
            print(f"\n--- Dataset: {dataset_name} ---")
            
            search_pattern = os.path.join(base_dir, dataset_name, "images", f"*.{ext}")
            image_paths = glob.glob(search_pattern)
            
            if not image_paths:
                continue
                
            for T in TEMPERATURES:
                ood_scores_list = []
                ind_scores_list = []
                
                for path in image_paths:
                    img_name = os.path.splitext(os.path.basename(path))[0]
                    logit_path = os.path.join(logits_base_dir, model_name, dataset_name, f"{img_name}.pt")
                    
                    if not os.path.exists(logit_path):
                        continue
                        
                    semantic_scores = torch.load(logit_path)
                    
                    pathGT = path.replace("images", "labels_masks")
                    base_path_gt, _ = os.path.splitext(pathGT)
                    pathGT = base_path_gt + ".png"
                    
                    mask_pil = Image.open(pathGT)
                    ood_gts = np.array(mask_pil)
                    
                    if dataset_name == "RoadAnomaly":
                        raw = ood_gts.copy()
                    
                        mapped = np.full_like(raw, 255)
                        mapped[raw == 0] = 0      # in-distribution / valid background 
                        mapped[raw == 2] = 1      # anomaly / OOD
                    
                        ood_gts = mapped
                        
                    ood_mask = (ood_gts == 1)
                    ind_mask = (ood_gts == 0)
                    
                    anomaly_result = compute_anomaly_scaled(semantic_scores, T=T)
                    
                    ood_scores_list.append(anomaly_result[ood_mask].astype(np.float32))
                    ind_scores_list.append(anomaly_result[ind_mask].astype(np.float32))
                
                if ood_scores_list and len(np.concatenate(ood_scores_list)) > 0:
                    ood_out = np.concatenate(ood_scores_list)
                    ind_out = np.concatenate(ind_scores_list)
                    
                    val_out = np.concatenate((ind_out, ood_out))
                    val_label = np.concatenate((np.zeros(len(ind_out), dtype=np.uint8), np.ones(len(ood_out), dtype=np.uint8)))
                    
                    prc_auc = average_precision_score(val_label, val_out)
                    fpr = fpr_at_95_tpr(val_out, val_label)
                    
                    print(f"  T = {T:4.2f} | MSP -> AUPRC: {prc_auc*100:.2f}% | FPR@TPR95: {fpr*100:.2f}%")

if __name__ == '__main__':
    main()