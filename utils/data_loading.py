import importlib


def build_datamodule(
    config,
    data_path="/content/drive/MyDrive/project/",
    batch_size=1,
    img_size=(1024, 1024),
    num_workers=0,
):
    """
    Build and initialize the datamodule defined in the YAML configuration.

    The datamodule class is dynamically imported from
    config["data"]["class_path"]. Additional initialization arguments are read
    from config["data"]["init_args"].
    """
    module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
    datamodule_cls = getattr(importlib.import_module(module_name), class_name)
    datamodule_kwargs = config["data"].get("init_args", {})

    datamodule = datamodule_cls(
        path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        check_empty_targets=False,
        **datamodule_kwargs,
    )

    datamodule.setup()
    return datamodule
    
