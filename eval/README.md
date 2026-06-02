# Anomaly Segmentation Evaluation

This folder contains the evaluation scripts used for road anomaly segmentation experiments.  
The current evaluation pipeline is organized around three main modules:

- `evalAnomalyEOMT.py`: anomaly segmentation evaluation for EoMT models.
- `evalAnomalyERFNET.py`: anomaly segmentation evaluation for the ERFNet baseline.
- `temperature.py`: temperature-scaling experiments for MSP-based anomaly scoring using saved semantic logits.

The scripts evaluate whether a semantic segmentation model can assign high anomaly scores to out-of-distribution pixels and low anomaly scores to in-distribution road-scene pixels.

## Requirements

The code was mainly used in a Colab-style environment. Exact versions may vary, but the following packages are required:

- Python 3.x
- PyTorch
- torchvision
- numpy
- Pillow
- scikit-learn
- PyYAML
- matplotlib
- `ood_metrics`
- CUDA-enabled GPU recommended for EoMT evaluation

For the legacy ERFNet/Cityscapes utilities, `visdom` is only required when using the optional `--visualize` flag.

## Dataset structure

The anomaly evaluation scripts expect a root folder containing the validation datasets, each with the following structure:

```text
Validation_Dataset/
├── FS_LostFound_full/
│   ├── images/
│   └── labels_masks/
├── RoadAnomaly/
│   ├── images/
│   └── labels_masks/
├── RoadAnomaly21/
│   ├── images/
│   └── labels_masks/
├── RoadObsticle21/
│   ├── images/
│   └── labels_masks/
└── fs_static/
    ├── images/
    └── labels_masks/
```

By default, the anomaly labels are interpreted as follows:

- `0`: in-distribution pixels
- `1`: anomaly / out-of-distribution pixels
- `255`: ignored pixels

For `RoadAnomaly`, labels are remapped internally so that `0` corresponds to in-distribution pixels and `2` corresponds to anomaly pixels.

## Main scripts

## `evalAnomalyEOMT.py`

This is the main evaluation script for EoMT-based models. It supports three model variants:

- `cityscapes`
- `coco`
- `coco_finetuned`

The script loads the selected EoMT configuration and checkpoint, performs semantic inference on each anomaly dataset, computes pixel-wise anomaly scores, and reports:

- AUPRC
- FPR@TPR95

The evaluated anomaly scoring methods are:

- `MSP`: `1 - max(softmax(score))`
- `MaxLogit`: `-max(score)`
- `MaxEntropy`: entropy of the softmax distribution
- `RbA`: `-sum(tanh(score))`

### Example

```bash
python evalAnomalyEOMT.py \
  --base_dir /content/drive/MyDrive/project/Anomaly_Validation_Datasets/Validation_Dataset \
  --model cityscapes
```

To evaluate the COCO-pretrained checkpoint:

```bash
python evalAnomalyEOMT.py \
  --base_dir /content/drive/MyDrive/project/Anomaly_Validation_Datasets/Validation_Dataset \
  --model coco
```

To evaluate the COCO-to-Cityscapes fine-tuned checkpoint:

```bash
python evalAnomalyEOMT.py \
  --base_dir /content/drive/MyDrive/project/Anomaly_Validation_Datasets/Validation_Dataset \
  --model coco_finetuned
```

### Saving semantic scores

The `--save_inf` flag saves the semantic score tensors to disk. This is useful when running `temperature.py`, because the temperature-scaling script reuses the saved logits instead of recomputing inference.

```bash
python evalAnomalyEOMT.py \
  --base_dir /content/drive/MyDrive/project/Anomaly_Validation_Datasets/Validation_Dataset \
  --model cityscapes \
  --save_inf
```

Saved tensors are stored under:

```text
/content/drive/MyDrive/project/saved_logits/<model_name>/<dataset_name>/<image_name>.pt
```

## `evalAnomalyERFNET.py`

This script evaluates the ERFNet baseline on the same anomaly validation datasets.

It loads the ERFNet architecture and checkpoint, computes the segmentation logits, applies post-hoc anomaly scoring, and reports:

- AUPRC
- FPR@TPR95

The evaluated anomaly scoring methods are:

- `MSP`
- `MaxLogit`
- `MaxEntropy`

Unlike the EoMT script, `evalAnomalyERFNET.py` uses the raw ERFNet output logits directly.

### Example

```bash
python evalAnomalyERFNET.py \
  --base_dir /content/drive/MyDrive/project/Anomaly_Validation_Datasets/Validation_Dataset \
  --loadDir ../trained_models/ \
  --loadWeights erfnet_pretrained.pth \
  --loadModel erfnet.py
```

Results are appended to:

```text
results.txt
```

## `temperature.py`

This module performs a temperature-scaling sweep for MSP-based anomaly detection.

It assumes that semantic score tensors have already been saved by running `evalAnomalyEOMT.py` with `--save_inf`. For each temperature value, it computes:

```text
MSP_T = 1 - max(softmax(scores / T))
```

and reports:

- AUPRC
- FPR@TPR95

The current script tests the following temperature values:

```python
[0.25, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0]
```

---

## Legacy Cityscapes utilities

The repository also contains legacy utilities inherited from the ERFNet evaluation code. These are not the main anomaly-evaluation scripts, but they may still be useful for debugging or visualization.

### `eval_cityscapes_color.py`

Produces colorized semantic segmentation predictions on Cityscapes images and saves them under:

```text
save_color/
```

Example:

```bash
python eval_cityscapes_color.py \
  --datadir /home/datasets/cityscapes/ \
  --subset val
```

### `eval_cityscapes_server.py`

Produces Cityscapes predictions converted back to original `labelIds`, suitable for evaluation with the official Cityscapes scripts or for submission to the Cityscapes server.

Outputs are saved under:

```text
save_results/
```

Example:

```bash
python eval_cityscapes_server.py \
  --datadir /home/datasets/cityscapes/ \
  --subset val
```

### `eval_iou.py`

Computes mean IoU and per-class IoU on labeled Cityscapes subsets.

Example:

```bash
python eval_iou.py \
  --datadir /home/datasets/cityscapes/ \
  --subset val
```

### `eval_forwardTime.py`

Measures ERFNet forward-pass time at a given resolution.

Example:

```bash
python eval_forwardTime.py \
  --width 1024 \
  --height 512
```