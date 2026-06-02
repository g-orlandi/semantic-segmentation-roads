# Road Scene Segmentation and Anomaly Detection with EoMT and ERFNet

This repository contains the code used for semantic segmentation and anomaly segmentation experiments on road-scene datasets.
The project focuses on evaluating mask-based segmentation architectures, in particular EoMT and comparing them with an ERFNet baseline for road anomaly detection.

The repository was developed starting from the project template for:

* Mask Architecture Anomaly Segmentation for Road Scenes
* Comprehensive Road Scene Understanding for Autonomous Driving

The final codebase includes both the original ERFNet evaluation utilities and the EoMT-based pipeline used for semantic segmentation, fine-tuning, post-hoc anomaly scoring and temperature-scaling experiments.

## Repository structure

```text
semantic-segmentation-roads/
├── eomt/
├── eval/
├── utils/
├── trained_models/
├── notebooks/
├── results/
└── README.md
```

## Main folders

### `eomt/`

Contains the EoMT codebase used for model definition, training, fine-tuning and semantic evaluation.

Important files:

* `eomt/main.py`: training and evaluation entry point from the EoMT codebase.
* `eomt/semantic_eval.py`: semantic segmentation evaluation utilities.
* `eomt/configs/eomt_base_640_cs.yaml`: EoMT configuration for Cityscapes.
* `eomt/configs/eomt_base_640_coco.yaml`: EoMT configuration for COCO.
* `eomt/models/`: EoMT model implementation.
* `eomt/training/`: Lightning modules, losses and training utilities.

### `eval/`

Contains the anomaly segmentation evaluation scripts and legacy ERFNet evaluation tools.

Main project scripts:

* `evalAnomalyEOMT.py`: anomaly segmentation evaluation for EoMT.
* `evalAnomalyERFNET.py`: anomaly segmentation evaluation for ERFNet.
* `temperature.py`: temperature-scaling sweep for MSP anomaly scoring using saved semantic scores.

See `eval/README.md` for the detailed documentation of these scripts.

### `utils/`

Contains helper functions used by the EoMT evaluation pipeline.

Important files:

* `model_loading.py`: builds EoMT models and loads checkpoints.
* `label_mapping.py`: maps or aggregates COCO semantic scores into the Cityscapes label space.
* `data_loading.py`: dataset and dataloader utilities.

### `trained_models/`

Contains the ERFNet checkpoints used for the baseline evaluation.

### `notebooks/`

Contains experimental notebooks used during development, debugging, semantic evaluation, fine-tuning analysis, anomaly evaluation and visualization.

### `results/`

Contains saved experimental outputs, including semantic segmentation results, fine-tuning metrics, anomaly segmentation results and temperature-scaling results.

## Datasets

### Cityscapes

Cityscapes is used as the in-distribution semantic segmentation dataset.

Expected structure:

```text
cityscapes/
├── leftImg8bit/
└── gtFine/
```

### Anomaly validation datasets

The anomaly segmentation scripts expect a validation root folder with the following structure:

```text
Validation_Dataset/
├── FS_LostFound_full/
├── RoadAnomaly/
├── RoadAnomaly21/
├── RoadObsticle21/
└── fs_static/
```

## Installation

A GPU-enabled Python environment is recommended.

Install the EoMT requirements:

```bash
cd eomt
pip install -r requirements.txt
```

Additional packages used by the evaluation scripts include:

```bash
pip install numpy pillow scikit-learn matplotlib pyyaml torchvision
```

The project also expects the `ood_metrics` utilities to be available in the Python path.

## Semantic segmentation evaluation

Semantic segmentation experiments are mainly performed through the EoMT code and the notebooks under `notebooks/step_4/`.

A typical workflow is:

1. Load the EoMT configuration.
2. Load one of the model checkpoints.
3. Run semantic inference on Cityscapes validation images.
4. Compute mIoU.
5. Compare Cityscapes, COCO, and COCO-fine-tuned variants.

For the COCO checkpoint, predictions must be mapped or aggregated into the Cityscapes label space before comparison with Cityscapes labels.

## Anomaly segmentation evaluation

The anomaly segmentation evaluation is performed in the `eval/` folder.

### EoMT anomaly evaluation

```bash
cd eval

python evalAnomalyEOMT.py \
  --base_dir /content/drive/MyDrive/project/Anomaly_Validation_Datasets/Validation_Dataset \
  --model cityscapes
```

Available model options:

```text
cityscapes
coco
coco_finetuned
```

### ERFNet anomaly evaluation

```bash
cd eval

python evalAnomalyERFNET.py \
  --base_dir /content/drive/MyDrive/project/Anomaly_Validation_Datasets/Validation_Dataset \
  --loadDir ../trained_models/ \
  --loadWeights erfnet_pretrained.pth \
  --loadModel erfnet.py
```

## Temperature-scaling experiments

Temperature scaling is evaluated through `eval/temperature.py`.

First, save EoMT semantic scores:

```bash
cd eval

python evalAnomalyEOMT.py \
  --base_dir /content/drive/MyDrive/project/Anomaly_Validation_Datasets/Validation_Dataset \
  --model cityscapes \
  --save_inf
```

Then run:

```bash
python temperature.py
```

## Evaluation metrics

- Semantic segmentation: **mIoU**
- Anomaly segmentation: **AUPRC** and **FPR@TPR95**

This aggregation is applied before computing post-hoc anomaly scores for the COCO checkpoint.

## Reproducibility notes

Several scripts contain hard-coded paths from the original Colab/Drive setup.
Before running the code on a different machine, update:

* dataset paths,
* checkpoint paths,
* saved logits paths,
* result output paths.

The main hard-coded paths are in:

* `eval/evalAnomalyEOMT.py`
* `eval/evalAnomalyERFNET.py`
* `eval/temperature.py`

