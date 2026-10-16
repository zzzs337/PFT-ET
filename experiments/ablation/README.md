# Ablation experiments

Full model plus ten configs in `configs/ablation`. Group A: current-only, history, future-distillation, deterministic future. Group B: single Gaussian, top-1, mean-only, without transition, without EMA, without auxiliary. `scripts/run_experiments.py` trains three runs per config and aggregates results. Undefined history/distillation details are recorded in implementation notes.
