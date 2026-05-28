import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from torchmetrics.classification import MulticlassJaccardIndex
from torch.nn import functional as F
from tqdm.notebook import tqdm

from utils.label_mapping import get_coco_to_cityscapes_mapping


IGNORE_INDEX = 19   # pixels not covered by any GT mask, and COCO classes not mapped to Cityscapes


def infer_semantic(model, img, target, device):
    """
    Run semantic inference on a single image.

    Returns:
        pred_array: (H, W) array with predicted class indices in the model's native space.
        target_array:   (H, W) array with Cityscapes train IDs [0-18] or IGNORE_INDEX.
    """
    with torch.no_grad(), autocast(dtype=torch.float16, device_type='cuda'):
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:] for img in imgs]

        crops, origins = model.window_imgs_semantic(imgs)

        mask_logits_per_layer, class_logits_per_layer = model(crops)
        
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], model.img_size, mode='bilinear'
        )
        crop_logits = model.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1]
        )
        
        logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        
        pred_array = logits[0].argmax(0).cpu().numpy()

    target_array = model.to_per_pixel_targets_semantic([target], IGNORE_INDEX)[0].cpu().numpy()
    return pred_array, target_array

def make_metric(device):
    """
    Create the per-class IoU metric used for semantic segmentation evaluation.
    """
    return MulticlassJaccardIndex(
        num_classes=20,
        ignore_index=IGNORE_INDEX,
        average=None,          # returns per-class IoU
        validate_args=False,
    ).to(device)

def evaluate_semantic(model, val_loader, device, coco=False, limit_batches=None):
    """
    Evaluate semantic segmentation using mIoU.

    If coco=True, predictions are remapped from selected COCO class IDs
    to Cityscapes train IDs before computing the metric.
    """
    model.eval()

    if coco:
        coco_to_cityscapes = get_coco_to_cityscapes_mapping()

    metric = make_metric(device)

    total = limit_batches if limit_batches else len(val_loader)
    print(f'Evaluating on {total} images...')

    for batch_idx, batch in enumerate(tqdm(val_loader, total=total, desc='Eval')):
        if limit_batches and batch_idx >= limit_batches:
            break

        imgs, targets = batch
        for img, target in zip(imgs, targets):
            pred, gt = infer_semantic(model, img, target, device)
            if coco:
                pred = coco_to_cityscapes[pred]
            # GT is already in [0..18] plus {IGNORE_INDEX}.
            metric.update(
                torch.from_numpy(pred.astype(np.int64)).to(device),
                torch.from_numpy(gt.astype(np.int64)).to(device)
            )

    iou_per_class = metric.compute()[:19].cpu().numpy()  # only first 19 classes
    miou = float(iou_per_class.mean())
    
    print(f'\nmIoU: {miou * 100:.2f}%')
    return miou