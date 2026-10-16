# Baseline adaptation and validation-only tuning

| Method | Input adaptation | Event risk |
| --- | --- | --- |
| TFD | RGB sampling within the 40-frame observation span | Binary fall score |
| TCNTE | COCO-17 skeleton observations | Fall score |
| Pose-ViT | COCO-17 pose observations | Fall score |
| DGNN-LSTM | Observed skeleton and predicted motion | Fall score |
| Dynamic-DeepHit | Longitudinal current-state features; six finite bins plus tail | Sum of six finite-bin probabilities |

All predictions must cover the same protocol-valid sequence/frame positions. The evaluator supplies labels from the manifest rather than trusting external labels. Neither future observations nor event annotations are model inputs.

## Search and final runs

`configs/baselines.yaml` configures six candidates: three learning rates times two weight decays, one selection run per candidate, at most 120 epochs, batch size 16, then three final seeds. Completed runs record their selected configuration and validation outcomes. The registry is marked as an execution plan; historical search logs have not been verified.

A baseline adapter is a Python module with these callables:

```python
def fit(config, params, seed, output):
    # Train on config's training partition and return a checkpoint path.
    ...

def predict(config, checkpoint, partition, output):
    # Write JSONL predictions for every valid prefix; return the output path.
    ...
```

The adapter retains the baseline architecture and implements its native input processing within the shared observation span. Record the upstream source commit, input sampler, output-head definition, optimizer, epochs completed, and weight provenance in the adapter's training directory.

```bash
python scripts/run_baseline_search.py --config configs/main/upfall.yaml --method tfd --adapter /path/to/tfd_adapter.py --output outputs/tfd/search
python scripts/run_baseline_search.py --config configs/main/upfall.yaml --method tfd --adapter /path/to/tfd_adapter.py --output outputs/tfd/final --final-from outputs/tfd/search/selected.json
```

The adapter path is an input supplied with the corresponding baseline implementation. Search inputs contain training and validation membership only. Each candidate saves parameters, seed, checkpoint hash, validation predictions, all 99 threshold outcomes, and elapsed execution time. Selection maximizes validation Event F1, then minimizes FAR, then uses the declared candidate order. Final runs evaluate the test partition only after validation selection.

## Independent prediction evaluation

A JSONL row contains `sequence_id`, `frame`, `time`, and either `risk` in [0,1] or seven normalized `probability` values. If both outputs exist, they must agree.

```bash
python scripts/evaluate.py --config configs/main/upfall.yaml --partition val --predictions outputs/tfd/val_predictions.jsonl --output outputs/tfd/validation
python scripts/evaluate.py --config configs/main/upfall.yaml --partition test --predictions outputs/tfd/test_predictions.jsonl --threshold 0.73 --validation-record outputs/tfd/validation/val_evaluation_context.json --output outputs/tfd/test
```

Replace 0.73 with the threshold actually selected on validation. The test evaluator checks its validation record, data fingerprints, and seed. Scalar risks support event metrics and Brier/ECE; event-time NLL and remaining-time MAE require a full distribution.

## Evidence audit

After completing search and final runs for all five methods, run:

```bash
python scripts/audit_baseline_runs.py --registry configs/baselines.yaml --runs-root outputs/baselines --output outputs/baselines/audit.json
```

The audit checks the declared six-candidate budget, validation-only selection, three final seeds, adapter digest continuity, checkpoint presence, validation predictions, threshold records, and test evaluation records. It reports missing evidence without creating experimental results. The expected record structure is illustrated by `examples/baselines/`.
