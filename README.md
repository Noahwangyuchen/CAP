# CAP: Concept-Adaptive Pseudolabeling

Official implementation for **Handling Imbalanced Pseudolabels for Vision-Language Models with Concept Alignment and Confusion-Aware Calibrated Margin** (ICML 2025).

CAP adapts CLIP-style vision-language models to downstream image classification tasks with unlabeled data. The method targets imbalanced pseudolabels by addressing two failure modes:

- **Concept mismatch**: a class name is poorly aligned with the corresponding visual concept, so confident pseudolabels for that class are often wrong.
- **Concept confusion**: visually or semantically similar classes are difficult to separate, causing biased predictions and imbalanced pseudolabels.

CAP combines concept alignment with a confusion-aware calibrated margin, and trains independent visual adapters for reliable concept-aligned pseudolabels and dynamically generated unlabeled-data pseudolabels.

## Paper

- Paper: [PMLR](https://proceedings.mlr.press/v267/wang25f.html)
- OpenReview: [QIL44dSUPo](https://openreview.net/forum?id=QIL44dSUPo)
- Code: [github.com/Noahwangyuchen/CAP](https://github.com/Noahwangyuchen/CAP)

## Highlights

- Studies pseudolabel imbalance in VLM adaptation and attributes it to concept mismatch and concept confusion.
- Uses LLM-enhanced textual descriptions for detected concept-mismatched classes.
- Introduces a confusion-aware calibrated margin to encourage more discriminative and balanced predictions.
- Evaluated on **Flowers102**, **RESISC45**, **DTD**, **EuroSAT**, **CUB**, and **FGVCAircraft** under **UL**, **SSL**, and **TRZSL** settings.

## Environment

The original experiments use PyTorch 1.9.1 with CUDA 11.1.

```bash
git clone https://github.com/Noahwangyuchen/CAP.git
cd CAP

bash setup.sh
```

If you install dependencies manually, include the packages in `requirements.txt` plus the runtime packages used by the training code:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install scipy scikit-learn pytorch-metric-learning importlib_metadata
```

## Data

Place datasets under one root directory and pass that root through `DATASET_DIR`. The code expects each dataset to live at:

```text
${DATASET_DIR}/
  EuroSAT/
  DTD/
  RESICS45/
  FGVCAircraft/
  CUB/
  Flowers102/
```

Each dataset directory should contain the metadata files expected by `utils/prepare_data.py`, such as `train.txt`, `val.txt`, `test.txt`, `class_names.txt`, `labels.txt`, or the RESISC45 JSON split files, depending on the dataset. Class split metadata used by the project is stored in `data/data_splits/`, and class-name files are stored in `data/class_files/`.

Before running, create output directories:

```bash
mkdir -p logs pseudolabels evaluation trained_prompts saved_models
```

## Training

Training is launched through `accelerate`. The main entry point is `run_main.py`, and the learning paradigm is selected by the config passed to `--model_config`.

Set the run parameters:

```bash
export DATASET_DIR=/path/to/datasets
export DATASET_NAME=EuroSAT
export VIS_ENCODER='ViT-B/32'
export SPLIT_SEED=500
export OPTIM_SEED=1
export LR=0.01
export MARGIN_SCALE=12.0
```

Run semi-supervised learning (SSL):

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file methods_config/accelerate_localtest_config.yml \
  run_main.py --model_config ssl_config.yml
```

Run unsupervised learning (UL):

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file methods_config/accelerate_localtest_config.yml \
  run_main.py --model_config ul_config.yml
```

Run transductive zero-shot learning (TRZSL):

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file methods_config/accelerate_localtest_config.yml \
  run_main.py --model_config trzsl_config.yml
```

For multi-GPU runs, use `methods_config/accelerate_config.yml` or edit the Accelerate config to match your machine. A sweep-style example is provided in `scripts/run.sh`.

## Outputs

The training pipeline writes:

- `results.json`: appended JSON lines with config and accuracy.
- `logs/`: per-run logs.
- `pseudolabels/`: cached CLIP pseudolabels and image/text features.
- `evaluation/`: pickled predictions, logits, labels, and text features.
- `trained_prompts/` and `saved_models/`: prompt/model checkpoints when enabled by the training code.

## Repository Structure

```text
CAP/
  run_main.py                 # main launcher
  methods/
    main.py                   # experiment workflow and argument parsing
    cap.py                    # CAP training dataset and loss wrapper
    cacm.py                   # confusion-aware calibrated margin logic
    training.py               # training and evaluation loops
  models/                     # CLIP/MaPLe prompt-learning modules
  custom_clip/                # local CLIP implementation
  data/
    dataset.py                # dataset wrappers
    dataset_prompts.py        # dataset-specific prompt templates
    text_augmentations.py     # enhanced descriptions for concept alignment
    class_files/              # class-name metadata
    data_splits/              # class split metadata
  methods_config/             # UL, SSL, TRZSL, and Accelerate configs
  utils/                      # metrics, pseudolabeling, schedulers, data helpers
```

## Citation

```bibtex
@inproceedings{
wang2025handling,
title={Handling Imbalanced Pseudolabels for Vision-Language Models with Concept Alignment and Confusion-Aware Calibrated Margin},
author={Yuchen Wang and Xuefeng Bai and Xiucheng Li and Weili Guan and Liqiang Nie and Xinyang Chen},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
}
```

## Acknowledgements

This repository builds on CLIP, prompt tuning, and pseudolabeling methods including FPL, GRIP, CPL, and MaPLe.
