# Metric definitions and units

The evaluator uses the following metric definitions and units.

| Metric | Unit | Denominator / calculation | Evidence source |
|---|---|---|---|
| Event F1 | fraction or percent | `2 TP / (2 TP + FP + FN)` from one-to-one fall/alarm matching; alarm onset must fall in `[T_F − 3s, T_F)` | Section 4.3, Table S1 |
| Confirmed alarm | event | Candidate if cumulative risk ≥ validation-selected threshold; confirmed when at least 3 of latest 5 positions are candidates; a contiguous confirmed interval counts once and new onset occurs only after reset | Eq. (19), Section 4.3 |
| FAR | independent confirmed ADL alarms per **camera-stream hour** | ADL alarm count / valid non-fall stream duration; simultaneous UP-Fall cameras count as separate video streams but not independent subjects | Sections 4.1, 4.3, Table S1 |
| Valid ADL exposure | hours | Each valid 20-FPS prediction position contributes 1/20 s on the original camera timeline; overlapping observation windows do not multiply duration | Section 4.3, Supplement A.1 |
| Completion ALT | s | Mean `(T_F − a)` over successfully matched falls; missing matches excluded from ALT but included in recall denominator | Section 4.3 |
| Impact-relative lead | s, signed | `T_impact − matched_alarm_onset`; zero or negative means not strictly pre-impact | Section 4.3 |
| Pre-impact Recall@L | fraction or percent | Number of evaluable falls with matched impact-relative lead ≥ L divided by **all** falls with evaluable impact boundaries, including misses; at L=0 require strictly positive lead | Figure 3(b), Table S3 |
| Event-time NLL | mean negative log probability per valid prefix | `−mean(log p_{i,t}[label])`, six half-second time bins + one tail | Section 4.3 |
| Remaining-time MAE | s | Mean absolute error using the non-tail probability-normalized bin-center estimate, only for valid fall prefixes within 3-s horizon | Section 4.3 |
| Candidate FPR | fraction or percent | Candidate-positive positions / all valid ADL positions within activity class; distinct from confirmed-event FAR | Section 4.7, Table 5 |
| 3-s Brier | unitless | Mean squared difference between 3-s cumulative predicted fall risk and binary indicator `label != tail` over the declared valid-prefix evaluation sample | Section 4.3, 4.6 |
| ECE | unitless | Ten equal-width predicted-risk bins; sum of absolute differences between mean predicted risk and observed 3-s incidence, weighted by number of valid prefixes in bin | Section 4.3, 4.6 |
| Transition RMSE | reference standardized latent units | Root of coordinate-averaged squared error of conditional mean; mixture conditional mean = weighted component means | Section 4.8, Table 6(a) |
| Transition NLL | negative log-density | Negative log mixture density of a 256-d reference target in shared fixed standardized coordinates; density sums coordinate log terms | Section 4.8, Table 6(a) |
| Coordinate coverage@q | percent | Mean fraction of target coordinates within the marginal q prediction interval, averaged over valid prefixes and dimensions | Section 4.8, Table S4 |
| Effective modes | count | `exp(−sum_k pi_k log(pi_k+epsilon))` | Eq. (20), Table 6(b) |
| Component overlap / separation / event-head JS | dimensionless | Pairwise Gaussian Bhattacharyya coefficient / pooled-covariance Mahalanobis mean separation / mixture-weighted divergence of component event-time outputs from marginalized event-time distribution | Section 4.8, Table 6(c) |
| Cluster bootstrap | percentile 95% CI | UP-Fall: resample subjects including both views as a cluster; UR Fall: resample complete sequences; recompute metrics at each replicate | Supplement A.2 |

## Important distinctions

- Event F1 `FP` can include unmatched alarms on fall streams; FAR numerator counts confirmed alarms **during ADL monitoring only**. They are not interchangeable.
- Three independent model seeds run on the same recordings do **not** increase independent observation time. FAR is separately calculated for each run and reported as an exposure-weighted mean of those run rates.
- A first-impact annotation shift of ±1 frame is ±0.05 s at 20 FPS; already-matched alarms remain fixed in the annotation sensitivity analysis.
- Validation chooses thresholds; test data are used only after thresholds have been frozen. Table 4 uses PFT-ET's validation FAR as a common budget, not a budget chosen on test data.
- Brier/ECE measure final event-risk probability, whereas latent-coordinate coverage measures the future-transition distribution. These do not substitute for one another.
- Reference-space comparisons use `scripts/reference_transition.py` and `scripts/evaluate_shared_transition.py`. Main-model EMA diagnostics are separately labeled.
- Cluster uncertainty uses linked per-event records and 10,000 shared subject/sequence bootstrap draws. Both cameras and all model seeds retain the same sampled cluster membership.
