import torch
from torch.amp.autocast_mode import autocast
from torchmetrics.classification import MulticlassJaccardIndex
from torch.nn import functional as F
from tqdm.notebook import tqdm
import numpy as np
import importlib
import yaml
from lightning import seed_everything

seed_everything(0, verbose=False)

DEBUG = True
IGNORE_INDEX = 19   # pixels not covered by any GT mask, and COCO classes not mapped to Cityscapes
LIMIT_BATCHES = None


def infer_semantic(model, img, target, device):
    """
        Returns:
        pred   (H, W) int  -- class indices in model's native space
        gt     (H, W) int  -- Cityscapes train_id [0-18] or IGNORE_INDEX
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

def load_weights(model, state_dict_path, device):
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    model_state_dict = model.state_dict()

    cleaned_state_dict = {}

    for k, v in state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            cleaned_state_dict[k] = v
        else:
            print(f"Ignored {k} (shape mismatch or not existing)")

    result = model.load_state_dict(cleaned_state_dict, strict=False)

    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)

    return model

def make_metric(device):
    return MulticlassJaccardIndex(
        num_classes=20,
        ignore_index=IGNORE_INDEX,
        average=None,          # returns per-class IoU
        validate_args=False,
    ).to(device)


def evaluate_semantic(model, val_loader, device, coco=False):
    if coco:
        coco_to_cityscapes = get_coco_to_cs()
    metric = make_metric(device)
    total = LIMIT_BATCHES if LIMIT_BATCHES else len(val_loader)
    print(f'Evaluating on {total} images...')

    for batch_idx, batch in enumerate(tqdm(val_loader, total=total, desc='Eval')):
        if LIMIT_BATCHES and batch_idx >= LIMIT_BATCHES:
            break

        imgs, targets = batch
        for img, target in zip(imgs, targets):
            pred, gt = infer_semantic(model, img, target, device)
            if coco:
                pred = coco_to_cityscapes[pred]
            # GT is already in [0..18] ∪ {IGNORE_INDEX}.
            metric.update(
                torch.from_numpy(pred.astype(np.int64)).to(device),
                torch.from_numpy(gt.astype(np.int64)).to(device)
            )

    iou_per_class = metric.compute()[:19].cpu().numpy()  # only first 19 classes
    miou = float(iou_per_class.mean())
    print(f'\nmIoU: {miou * 100:.2f}%')
    return miou

def get_config(coco=False):
    if coco:
        config_path = 'configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml'
    else:  
        config_path = 'configs/dinov2/cityscapes/semantic/eomt_base_640.yaml'
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_data(config, batch_size=1, img_size=(1024,1024)):

    data_path = "/content/drive/MyDrive/project/"

    data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
    data_module = getattr(importlib.import_module(data_module_name), class_name)
    data_module_kwargs = config["data"].get("init_args", {})

    data = data_module(
        path=data_path,
        batch_size=batch_size,
        num_workers=0,
        img_size=img_size,
        check_empty_targets=False,
        **data_module_kwargs
    ).setup()

    return data
    
def get_model(config, img_size, num_classes, coco=False, masked_attn_enabled=False):
    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    
    if coco:
        coco_data_init_args = config["data"].get("init_args", {})
        img_size = coco_data_init_args.get("img_size", (640, 640))
        num_classes = coco_data_init_args.get("num_classes", 133)

    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))
    
    # Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=masked_attn_enabled,
        num_classes=num_classes,
        encoder=encoder,
        **network_kwargs,
    )
    
    # Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    
    if coco:
        model_kwargs["stuff_classes"] = coco_data_init_args.get("stuff_classes", list(range(80, 133)))

    model = lit_cls(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        **model_kwargs,
    )
    return model
    
    
def get_coco_to_cs():
    # Inizializziamo a 19 (Ignore Index sicuro per torchmetrics)
    coco_to_cityscapes = np.full(256, 19, dtype=np.uint8)
    
    # Mappatura manuale derivata dalla ricerca sulle categorie
    coco_to_cityscapes[0] = 11   # person -> person
    coco_to_cityscapes[1] = 18   # bicycle -> bicycle
    coco_to_cityscapes[2] = 13   # car -> car
    coco_to_cityscapes[3] = 17   # motorcycle -> motorcycle
    coco_to_cityscapes[5] = 15   # bus -> bus
    coco_to_cityscapes[6] = 16   # train -> train
    coco_to_cityscapes[7] = 14   # truck -> truck
    coco_to_cityscapes[9] = 6    # traffic light -> traffic light
    coco_to_cityscapes[11] = 7   # stop sign -> traffic sign
    coco_to_cityscapes[82] = 2   # bridge -> building
    coco_to_cityscapes[90] = 9   # gravel -> terrain
    coco_to_cityscapes[91] = 2   # house -> building
    coco_to_cityscapes[100] = 0  # road -> road
    coco_to_cityscapes[101] = 2  # roof -> building
    coco_to_cityscapes[102] = 9  # sand -> terrain
    coco_to_cityscapes[109] = 3  # wall-brick -> wall
    coco_to_cityscapes[110] = 3  # wall-stone -> wall
    coco_to_cityscapes[111] = 3  # wall-tile -> wall
    coco_to_cityscapes[112] = 3  # wall-wood -> wall
    coco_to_cityscapes[116] = 8  # tree-merged -> vegetation
    coco_to_cityscapes[117] = 4  # fence-merged -> fence
    coco_to_cityscapes[119] = 10 # sky-other-merged -> sky
    coco_to_cityscapes[123] = 1  # pavement-merged -> sidewalk
    coco_to_cityscapes[125] = 8  # grass-merged -> vegetation
    coco_to_cityscapes[126] = 9  # dirt-merged -> terrain
    coco_to_cityscapes[129] = 2  # building-other-merged -> building
    coco_to_cityscapes[131] = 3  # wall-other-merged -> wall
    return coco_to_cityscapes