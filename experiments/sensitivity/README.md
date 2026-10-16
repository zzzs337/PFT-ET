# Sensitivity experiments

K: 1,2,4,6,8; Hf: .5,1,1.5,2 seconds; N_MC: 1,4,8,16. Use the commands in README; select/report these comparisons on validation, as Sec. 4.9 specifies. Test-time sample counts reuse the same checkpoint; changing K or Hf retrains.

Additional split membership is generated only as an alternative experiment, with unchanged partition sizes. Use `scripts/make_splits.py` with three distinct explicit seeds. Create inherited configs that update only `data.splits` and `experiment.name` for full/current/deterministic strategies, then run `scripts/run_experiments.py --configs ...`. The original IDs must be separately supplied; generated splits cannot stand in for the paper's original partition. Keep each split's thresholds and test events separately. Do not mix split variability with optimization-seed SD.
