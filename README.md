# PFT-ET: Probabilistic Future-Transition Event-Time Prediction for Early Fall Warning

PFT-ET is a PyTorch framework for early fall warning from continuous RGB video and human skeleton sequences. It models possible future state transitions with a conditional Gaussian mixture and predicts a discrete event-time distribution within a finite horizon. The resulting risk sequence is converted into event-level warnings using temporal confirmation.

## Method overview

- RGB and skeleton inputs are sampled at 20 FPS with a 40-frame observation window.
- A multimodal encoder represents the observed motion state.
- A conditional Gaussian mixture predicts latent future transitions over a 1.5 s future window.
- The event head marginalizes the predicted transitions into six 0.5 s event-time intervals and a tail state beyond the 3 s horizon.
- Online alarms use a 3-of-5 confirmation rule. Alarm thresholds are selected on the validation set.

## Project structure

```text
.
├── configs/             # Main, ablation, sensitivity, and baseline settings
├── data/                # Dataset manifests, split files, and data documentation
├── experiments/         # Experiment entry points and protocol notes
├── pretrained/          # Pretrained-model resource instructions
├── scripts/             # Data preparation, training, inference, and evaluation
├── src/pft_et/          # Model, loss, data, metric, training, and utility modules
├── tests/               # Unit and execution tests
├── requirements.txt
├── requirements-backbone.txt
├── requirements-pose.txt
└── pyproject.toml
```

## Installation

Python 3.10 and PyTorch 2.5.1 are used for the main experiments.

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt -r requirements-backbone.txt
pip install -e . --no-deps
```

Pose extraction uses the dependencies listed in `requirements-pose.txt`. Install the corresponding PyTorch, CUDA, MMCV, MMEngine, and MMPose builds in a separate preprocessing environment.

## Data preparation

Experiments use the UP-Fall and UR Fall datasets. Obtain the datasets from their original sources and follow `data/README.md` for the manifest, annotation, skeleton, and split schemas. UP-Fall is divided at the subject level, and UR Fall is divided at the complete-sequence level before prefix generation. The two UP-Fall RGB views are processed as separate camera streams; UR Fall uses Camera 0.

Prepare the RGB and skeleton arrays at 20 FPS:

```bash
python scripts/extract_pose.py \
  --manifest data/raw/upfall/manifest.jsonl \
  --pose-config /path/to/rtmpose_config.py \
  --pose-checkpoint /path/to/rtmpose_checkpoint.pth \
  --device cuda

python scripts/prepare_data.py \
  --manifest data/raw/upfall/manifest.jsonl \
  --output data/processed/upfall

python scripts/prepare_data.py \
  --manifest data/raw/urfall/manifest.jsonl \
  --output data/processed/urfall
```

Set the processed manifest and split paths in the corresponding YAML configuration.

## Training

```bash
python scripts/train.py --config configs/main/upfall.yaml
python scripts/train.py --config configs/main/urfall.yaml
```

Resume a run from a saved checkpoint:

```bash
python scripts/train.py \
  --config configs/main/upfall.yaml \
  --resume outputs/upfall/pft_et/seed2024/last.pt
```

Each run stores the resolved configuration, selected validation threshold, checkpoints, and training log in its configured output directory.

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/main/upfall.yaml \
  --checkpoint outputs/upfall/pft_et/seed2024/best.pt \
  --analysis
```

Evaluation applies the same 3-of-5 temporal confirmation and one-to-one event-matching rules to all methods. The main measures are Event F1, false alarms per camera-stream hour, and warning lead time. Event-Time NLL, remaining-time MAE, pre-impact recall, transition statistics, and risk-calibration measures are produced when the required predictions and annotations are available.

## Inference

```bash
python scripts/inference.py \
  --config configs/main/upfall.yaml \
  --checkpoint outputs/upfall/pft_et/seed2024/best.pt \
  --input data/processed/upfall/manifest.jsonl \
  --output outputs/inference.jsonl
```

The output contains the event-time distribution, cumulative 3 s fall risk, expected remaining time, candidate state, confirmed state, and alarm onset for each valid prediction position.

## Main experiments

Run the three configured seeds for both datasets and aggregate their results:

```bash
python scripts/reproduce_main_results.py
```

Evaluate existing checkpoints without retraining:

```bash
python scripts/reproduce_main_results.py --eval-only
```

Experiment configurations for ablation and sensitivity analyses are provided under `configs/ablation/` and `configs/sensitivity/`.

## Baseline evaluation

Baseline methods use the same data partitions, event boundaries, 40-frame observation window, temporal confirmation, and event-matching protocol. External predictions can be supplied as a scalar `risk` for each valid prefix or as a seven-state event-time `probability` vector. Convert tabular baseline output with:

```bash
python scripts/convert_baseline_predictions.py \
  --input outputs/baseline/raw_predictions.csv \
  --output outputs/baseline/val_predictions.jsonl \
  --score-field fall_probability \
  --score-type probability
```

Select the alarm threshold on validation predictions and reuse it for test evaluation:

```bash
python scripts/evaluate.py \
  --config configs/main/upfall.yaml \
  --partition val \
  --predictions outputs/baseline/val_predictions.jsonl \
  --output outputs/baseline/validation

python scripts/evaluate.py \
  --config configs/main/upfall.yaml \
  --partition test \
  --predictions outputs/baseline/test_predictions.jsonl \
  --threshold VALIDATION_THRESHOLD \
  --output outputs/baseline/test
```

## Tests

```bash
pytest -q tests/
python scripts/smoke_test.py --output outputs/smoke
```

## Data terms

UP-Fall and UR Fall are third-party research datasets. Their original access conditions and licences apply. Pretrained backbones and pose-estimation weights remain subject to the terms of their respective sources.

## Citation

Please cite the associated article:

*PFT-ET: Probabilistic Future-Transition Event-Time Prediction for Early Fall Warning.*
