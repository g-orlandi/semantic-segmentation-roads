import importlib
import torch
import yaml
from pathlib import Path


def load_weights(model, state_dict_path, device):
    """
    Load compatible weights into a model.

    Only parameters whose names exist in the current model and whose shapes
    match are loaded. This is useful when loading checkpoints with partially
    different heads or class dimensions.
    """
    
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    model_state_dict = model.state_dict()

    cleaned_state_dict = {}

    for name, weights in state_dict.items():
        if name in model_state_dict and weights.shape == model_state_dict[name].shape:
            cleaned_state_dict[name] = weights
        else:
            print(f"Ignored {name} (shape mismatch or not existing)")

    result = model.load_state_dict(cleaned_state_dict, strict=False)

    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)

    return model


def get_config(coco=False):
    """
    Load the YAML configuration for either the Cityscapes or COCO model.
    """
    project_root = Path(__file__).resolve().parents[1]

    if coco:
        config_path = project_root / "eomt" / "configs" / "eomt_base_640_coco.yaml"
    else:
        config_path = project_root / "eomt" / "configs" / "eomt_base_640_cs.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def build_model(config, img_size, num_classes, coco=False, masked_attn_enabled=True):
    """
    Build the EoMT Lightning model from a YAML configuration.

    The encoder, network, and Lightning module classes are dynamically imported
    from their class paths in the config.

    When coco=True, image size, number of classes and stuff classes are read from the COCO data config.
    """
    
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