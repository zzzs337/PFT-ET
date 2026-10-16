# Implementation details

The main model uses the three-term objective in Eq. (17): event-time cross-entropy, transition negative log-likelihood, and auxiliary cross-entropy. No ranking, hard-ADL, or structural losses are added.

| Paper element | Implementation |
| --- | --- |
| Valid event/future prefixes, Eqs. (4), (6) | `data/dataset.py` |
| RGB-skeleton encoding and scalar fusion, Eq. (5) | `models/encoder.py` |
| EMA and stop-gradient target, Eqs. (7), (8) | `models/model.py` |
| Conditional Gaussian mixture and density, Eqs. (9), (10) | `models/transition.py` |
| Input [z || r] and two-stage marginalization, Eqs. (11)-(14) | `models/event_time.py` |
| Three training losses, Eqs. (15)-(17) | `losses/losses.py` |
| Risk and confirmation, Eqs. (18), (19) | `metrics/alarms.py`, `metrics/evaluation.py` |
| Fixed-coordinate controlled experiment | `scripts/reference_transition.py` |

## Numerical conventions

Python states are indexed 0–5 and tail=6. Exactly 0.5 s belongs to the first bin; exactly 3 s belongs to the last finite bin. Current and future modalities share eight temporal indices. Future targets use frames t+1 through t+30 and may cross completion. An unavailable future target masks transition supervision without removing the event-time example.

The event head takes the current state and the transition, not the future state z+r. The mixture NLL sums log densities across latent coordinates before log-sum-exp across components. Remaining time is conditional on a non-tail event and uses normalized finite-bin probabilities.

## Configurable implementation choices

The fusion gate is linear; state and prediction mappings use linear/GELU/dropout blocks. Skeleton positional encodings are sinusoidal. Coordinates use x/width and y/height with confidence retained. RGB normalization follows the configured pretrained backbone. These choices are explicit in the model modules and configuration.

The execution defaults for three runs are 2024, 2025, and 2026; actual original run identities must be supplied to reproduce historical experiments. History encoder and future-distillation architectures follow the existing package and remain configurable.

Pairwise component statistics use normalized pi_i*pi_j weights over unordered pairs. Separation is the pooled-covariance Mahalanobis distance without dimension normalization. JS is the mixture-weighted symmetric divergence between each component event distribution and the marginalized distribution.

## Coordinate systems

Main-model transition diagnostics use the checkpoint's EMA space. The controlled fixed-reference experiment instead freezes an explicitly identified online/target encoder, exports all partitions, fits normalization on training targets only, and trains separate transition heads against identical targets. These metrics are not interchanged. Scaling experiments multiply training targets by the declared factor and divide predicted means and scales by it before common-coordinate density and coverage evaluation.
