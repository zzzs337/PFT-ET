# Reproducibility procedures

Run commands from the project root. Use fresh output directories for new experiments. Execution defaults and implementation choices are recorded in `configs/default.yaml` and `implementation_notes.md`.

## Evaluation and independent recalculation

```bash
python scripts/evaluate.py --config configs/main/upfall.yaml --checkpoint outputs/upfall/pft_et/seed2024/best.pt --analysis --output outputs/upfall/evaluation/seed2024
python scripts/recompute_evaluation.py --run outputs/upfall/evaluation/seed2024 --output outputs/upfall/evaluation/seed2024/recomputed.json
python scripts/export_evaluation_records.py --runs outputs/upfall/evaluation/seed2024 --output outputs/upfall/event_tables
```

The independent recalculation loads saved predictions, metadata, configuration, and the fixed threshold. It checks exact valid-prefix membership, F1/FAR/ALT, pre-impact recall, calibration, alarm/reset traces, and exposure reconciliation without loading model weights or original arrays. Edited metrics and mismatched traces fail the check.

The warning timestamp is the first confirmed alarm position, not the first candidate. Confirmation waiting is therefore included in lead time. Computational latency is recorded separately and is not subtracted. Matching uses the earliest unmatched onset in `[completion − 3 s, completion)`. Persistent alarms retain their original onset. An ended alarm can be followed by a new eligible alarm after reset.

## Fixed reference encoder and likelihoods

Choose the actual reference checkpoint and explicitly identify its online or EMA target encoder. The checkpoint must use the same manifest and split as the experiment.

```bash
python scripts/reference_transition.py export --config configs/main/upfall.yaml --checkpoint outputs/upfall/pft_et/seed2024/best.pt --encoder target --output outputs/reference/seed2024
python scripts/reference_transition.py fit --reference outputs/reference/seed2024 --kind mixture --scale 1.0 --seed 2024 --output outputs/reference_heads/mixture/seed2024
python scripts/reference_transition.py predict --reference outputs/reference/seed2024 --checkpoint outputs/reference_heads/mixture/seed2024/best.pt --partition test --output outputs/reference_heads/mixture/seed2024/test.jsonl
python scripts/evaluate_shared_transition.py --targets outputs/reference/seed2024/test.jsonl --predictions outputs/reference_heads/mixture/seed2024/test.jsonl --output outputs/reference_heads/mixture/seed2024/metrics.json
```

Repeat `fit`/`predict` for `--kind deterministic` and `--kind single`, using the same reference export. For scale sensitivity, fit separate mixture heads with `--scale 0.5`, `1.0`, and `2.0`. Repeat the controlled fits with each declared training seed. This experiment trains transition heads on frozen features; it does not replace the main model's EMA training objective.

The exporter computes both current and future features with the same frozen encoder. Only valid transition targets are retained. State and transition normalization use training means and population standard deviations with a `1e-6` floor. Validation and test use the same parameters. The fitted head checkpoint records the reference and normalization hashes. Scaled predictions are returned to standardized reference coordinates before density evaluation. Gaussian NLL is joint over latent coordinates, with stable log-sum-exp across components; coordinate-wise mixture intervals use marginal CDF inversion.

## Case selection and interventions

```bash
python scripts/select_cases.py --features outputs/reference/seed2024/test.jsonl --predictions outputs/upfall/evaluation/seed2024/test_predictions.jsonl --neighbors 1 --top 1 --output outputs/cases/seed2024
python scripts/run_interventions.py --config configs/main/upfall.yaml --checkpoint outputs/upfall/pft_et/seed2024/best.pt --output outputs/interventions/seed2024
```

`--candidates` optionally supplies a predefined JSONL list of sequence/frame IDs. The saved candidate-prefix list defines the complete selection scope. Current-state neighbors are retrieved across different physical sequences, not between synchronized views of one event, then ranked by observed future-transition distance. Distances are root-mean-square Euclidean distances in the exported training-standardized reference coordinates. Event-time predictions are attached after selection and are not ranking inputs. Reproducing an existing figure requires its original candidate pool, reference choice, and selected IDs.

Interventions retain one trained checkpoint. Mixture-mean replacement uses one deterministic mixture-weighted transition, unlike per-component mean-only marginalization. Shuffling transfers the predicted transition distribution from another physical sequence within the same partition, retaining the receiving current state. Donors are drawn without event labels using a seed-controlled physical-sequence mapping, and their prefix IDs are saved. Each intervention selects its threshold on validation and freezes it for test. These explicit choices should be matched to the original intervention protocol when comparing historical values.

## Baselines

See `baselines.md` for the adapter contract, input/output adaptations, search space, six-candidate budget, selection records, and final-seed execution.

## Runtime measurements

```bash
python scripts/benchmark_runtime.py --config configs/main/upfall.yaml --checkpoint outputs/upfall/pft_et/seed2024/best.pt --mode model --warmup 20 --repeats 100 --output outputs/timing/model
python scripts/benchmark_runtime.py --config configs/main/upfall.yaml --checkpoint outputs/upfall/pft_et/seed2024/best.pt --mode prepared --warmup 20 --repeats 100 --output outputs/timing/prepared
python scripts/benchmark_runtime.py --config configs/main/upfall.yaml --checkpoint outputs/upfall/pft_et/seed2024/best.pt --mode raw --raw-manifest data/raw/runtime/manifest.jsonl --pose-config pretrained/rtmpose_config.py --pose-checkpoint pretrained/rtmpose_m.pth --warmup 20 --repeats 100 --output outputs/timing/raw
```

Raw mode requires a single-stream manifest with `frame_list`, original timestamps, and optionally one tracked-person bbox per source frame. It reads each current RGB frame, runs RTMPose, resizes/normalizes inputs, updates a causal rolling RGB/pose buffer, evaluates the model, transfers risk to the host, and updates confirmation. History is cached causally; future frames are not read. Source image reading is included, while compressed-video decoding and network transport are outside this file-list boundary. All methods compared under this boundary should use the same source format, hardware, caching, batch size, and measurement loop.

Prepared mode includes prepared-array loading and window preprocessing but excludes pose extraction and original-video decoding. Model mode times forward inference on tensors already on device and excludes input reads, transfer, and confirmation. Its scope must not be reported as end-to-end latency. The historical-state ablation requires its own history-buffer timing adapter; the generic stage benchmark targets full/current/deterministic PFT-ET paths.

Timing uses `perf_counter_ns` and CUDA synchronization before and after stages and complete predictions. The raw table records every measured iteration; summaries include mean, sample SD, median, P95, and serialized throughput. Warmup predictions are excluded. Stage measurements include their stated synchronization/transfer boundaries; overall wall time need not equal a rounded sum of stage means. CPU generated-input checks are distinct from the paper's RTX 3090 results.

## Validation command

```bash
python scripts/update_file_index.py
python scripts/validate_package.py --root . --check-index
python scripts/verify_revision.py --output outputs/verification_2_3
```

The verification driver runs unit tests, a generated-input training/evaluation chain, independent recalculation, all three controlled predictor families, scale transforms, case retrieval, interventions, and CPU timing interfaces. Research benchmark execution additionally requires the inputs listed in `VERIFICATION.md`.
