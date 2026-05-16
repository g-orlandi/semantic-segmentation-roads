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
    """
    Compute pixel-wise anomaly score from raw logits.

    Args:
        logits: torch.Tensor of shape [C, H, W] (raw model output, single image)
        method: str, one of 'msp', 'maxlogit', 'maxentropy'

    Returns:
        anomaly_result: np.ndarray of shape [H, W], higher = more anomalous
    """
    logits_np = logits.data.cpu().numpy()  # [C, H, W]

    if method == 'maxlogit':
        # MaxLogit: anomaly score = negative of max raw logit
        anomaly_result = -np.max(logits_np, axis=0)

    elif method == 'msp':
        # MSP (Maximum Softmax Probability): anomaly score = 1 - max(softmax)
        # Use float64 for numerical stability
        logits_t = logits.unsqueeze(0).float()          # [1, C, H, W]
        softmax = F.softmax(logits_t, dim=1).squeeze(0) # [C, H, W]
        softmax_np = softmax.data.cpu().numpy()
        anomaly_result = 1.0 - np.max(softmax_np, axis=0)

    elif method == 'maxentropy':
        # Max Entropy: anomaly score = Shannon entropy of softmax distribution
        logits_t = logits.unsqueeze(0).float()          # [1, C, H, W]
        softmax = F.softmax(logits_t, dim=1).squeeze(0) # [C, H, W]
        softmax_np = softmax.data.cpu().numpy()
        # H(x) = -sum(p * log(p)), clamp to avoid log(0)
        softmax_np = np.clip(softmax_np, 1e-9, 1.0)
        entropy = -np.sum(softmax_np * np.log(softmax_np), axis=0)  # [H, W]
        anomaly_result = entropy

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: msp, maxlogit, maxentropy")

    return anomaly_result


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  # can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument(
        '--method',
        default='maxlogit',
        choices=['msp', 'maxlogit', 'maxentropy'],
        help="Anomaly scoring method: 'msp' (Maximum Softmax Probability), "
             "'maxlogit' (Maximum Logit Score), 'maxentropy' (Maximum Entropy). "
             "Default: maxlogit"
    )
    args = parser.parse_args()

    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)
    print(f"Anomaly scoring method: {args.method.upper()}")

    model = ERFNet(NUM_CLASSES)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights LOADED successfully")
    model.eval()

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        # input_transform already returns a [C, H, W] tensor via ToTensor()
        # unsqueeze(0) adds the batch dimension -> [1, C, H, W]
        images = input_transform(Image.open(path).convert('RGB')).unsqueeze(0).float()

        if not args.cpu:
            images = images.cuda()

        with torch.no_grad():
            result = model(images)  # [1, C, H, W]

        # Squeeze batch dim -> [C, H, W]
        logits = result.squeeze(0)

        # Compute anomaly score based on chosen method
        anomaly_result = compute_anomaly_score(logits, args.method)

        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

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

        if 1 not in np.unique(ood_gts):
            continue
        else:
            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_result)

        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write("\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'Method: {args.method.upper()}')
    print(f'AUPRC score: {prc_auc * 100.0:.2f}')
    print(f'FPR@TPR95:   {fpr * 100.0:.2f}')

    file.write(f'    Method: {args.method.upper()}   AUPRC score: {prc_auc * 100.0:.4f}   FPR@TPR95: {fpr * 100.0:.4f}')
    file.close()


if __name__ == '__main__':
    main()
