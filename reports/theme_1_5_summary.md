# Theme 1.5 — Extreme Class Imbalance: declared standard loss

**Hand-in:** comparison table + half-page summary declaring the standard loss.
Full table: `reports/theme_1_5_comparison.md`. Figures: `reports/figures/theme_1_5_*.png`.
33 experiments + 3 ablations, all through the shared harness, all reproducible from
`configs/*.yaml`.

---

## Declaration

> **The standard training recipe for Themes 1.1–1.4 is plain (unweighted) binary
> cross-entropy, with the decision threshold tuned on the validation fold.**
> **Add SMOTE — training fold only — when the positive rate is below ~1%.**
> **Do not use class weighting. Do not use focal loss with γ > 2.**

This is a negative result against the recommended approach, and it is stated as such:
neither focal loss nor cost-sensitive weighting beat plain BCE on validation.

## Evidence (selection on validation; test used only for confirmation)

Change in AUPRC versus that model's own plain-BCE run, same dataset, same architecture:

| recipe | mean Δ AUPRC | worst | best | cells won |
|---|---|---|---|---|
| class-weighted | **−0.0277** | −0.0926 | +0.0023 | 2 / 9 |
| focal γ=1 | −0.0056 | −0.0174 | +0.0010 | 1 / 3 |
| focal γ=2 | −0.0064 | −0.0173 | +0.0010 | 1 / 3 |
| focal γ=5 | **−0.0241** | −0.0485 | −0.0074 | 0 / 3 |
| **SMOTE** | **+0.0062** | −0.0129 | **+0.0567** | **3 / 6** |

SMOTE is the only recipe with a positive mean effect and the only one that wins at least
half its cells. The test fold reproduces the same ordering.

## Why class weighting fails — the threshold explains it

Median tuned threshold on validation:

| recipe | min | median | max |
|---|---|---|---|
| plain BCE | 0.225 | 0.285 | 0.599 |
| class-weighted | 0.511 | **0.781** | 0.998 |

Class weighting does not teach the model to separate fraud better — it inflates every
positive score. Tuning the threshold then simply undoes that inflation (0.285 → 0.781),
so the operating point ends up in the same place. What does *not* get undone is the
damage to ranking quality, which is what AUPRC measures. **The lever that makes a model
care about the rare class is the operating threshold, not the loss function** — and the
threshold is a free, post-hoc calibration step.

This also explains why the pipeline document's early evidence (recall 0.36 → 0.60 on UCI)
is real but not attributable to the loss: our plain-BCE XGBoost reaches recall 0.617 at a
tuned threshold of 0.225, without any imbalance handling at all.

## Where SMOTE earns its place

SMOTE's mean advantage is small because it is near-neutral on the two milder datasets and
large on the extreme one. The single biggest cell in the whole grid:

| PaySim, MLP | AUPRC | recall | F1 |
|---|---|---|---|
| plain BCE | 0.834 | 0.663 | 0.792 |
| **+ SMOTE** | **0.886** | **0.843** | **0.882** |

At 917:1 skew the minority class is too thin for the network to shape a decision boundary
around; synthetic interpolation fixes that. At 3.5:1 (UCI) it does nothing. Hence the
positive-rate trigger in the declaration rather than a blanket "always use SMOTE".

## Per-dataset winners (test fold)

| dataset | best recipe | AUPRC | recall |
|---|---|---|---|
| UCI (22.1% pos) | XGBoost + SMOTE | 0.5606 | 0.642 |
| IEEE-CIS (3.50% pos) | XGBoost, plain BCE | 0.8256 | 0.699 |
| PaySim (0.11% pos) | XGBoost + SMOTE | 0.9814 | 0.971 |

XGBoost beats the MLP on every dataset. For Themes 1.1–1.4, which are neural by
construction, the relevant comparison is MLP-vs-MLP — and there the same conclusion holds.

## Ablations — do these numbers mean what they appear to?

**1. IEEE-CIS: the mandated split inflates the score by roughly half.** Identical model and
recipe, only the split changes:

| split | AUROC | AUPRC | recall |
|---|---|---|---|
| stratified random (mandated, R2) | 0.9689 | 0.8136 | 0.684 |
| **entity-disjoint** | **0.8518** | **0.4664** | **0.226** |

42% of IEEE-CIS rows belong to a card that appears more than once, so on a random split
the model partly recognises the customer rather than the fraud. **This affects every theme
that uses IEEE-CIS.** Themes 1.1 and 1.2 must use `data/splits/ieee_grouped.npz`.
Recommendation: keep the mandated split for continuity with existing numbers, but report
the entity-disjoint number alongside it from now on.

**2. PaySim is genuinely easy — not a feature-engineering artefact.** Dropping the four
post-transaction balance columns costs only AUPRC 0.979 → 0.946. The near-perfect score
survives, so it is not leakage we introduced. It does mean PaySim has no headroom: plain
XGBoost already scores 0.98 with no imbalance handling, so **PaySim cannot discriminate
between recipes** and should not be cited as evidence for or against any of them.

**3. A bug was found and fixed mid-run.** The XGBoost+SMOTE path median-imputed NaNs in the
training fold only, leaving val/test with raw NaNs — a train/serve mismatch that collapsed
IEEE-CIS AUPRC to 0.499. Fixed to impute all folds with training-fold medians (R3); the
cell recovered to 0.8146. A control run isolating imputation from SMOTE
(`t15_ieee_xgboost_impute_only`, AUPRC 0.8196) confirms imputation alone costs ~0.006, so
the remaining gap is SMOTE's own small negative effect. All three SMOTE+XGBoost cells were
re-run; `results.csv` holds the corrected numbers.

## Caveats

- **Single seed (42).** The class-weighting penalty is consistent across 8 of 9 cells, so
  the direction is probably real, but the magnitudes are not yet confidence-bounded. The
  master benchmark calls for 5 seeds; run those before this declaration is frozen.
- Focal loss sits within ±0.02 AUPRC of plain BCE. Calling it "no better" is honest;
  calling it "worse" would be over-reading a single seed.
- The MLP is deliberately small (2 hidden layers). A larger network might respond
  differently to focal loss, though the direction of the class-weighting effect should not
  change, since that argument is about score inflation rather than capacity.

## What themes 1.1–1.4 should inherit

1. Train with **plain BCE**.
2. **Always tune the threshold on validation** — this is the step that actually buys recall.
3. Add **SMOTE on the training fold** only when the positive rate is under ~1%.
4. Report **AUPRC and recall-at-fixed-precision**, not AUROC alone — AUROC saturated above
   0.97 on PaySim for every recipe while AUPRC still ranged from 0.518 to 0.981.
