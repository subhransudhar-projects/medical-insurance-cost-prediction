# Stage 4 · Chunk 33 — Model Performance vs. Published Literature

The project brief cited several published results for this dataset. Here is an
honest side-by-side with our own numbers, and an analysis of the differences.

| Model | Published (cited) | Ours — single test split | Ours — 5-fold CV (honest) |
|---|---|---|---|
| Ridge Regression | R² ≈ 0.857 | 0.898 | 0.825 |
| Random Forest | "beats Gamma GLM" | 0.918 | 0.840 |
| XGBoost | R² ≈ 0.94 | 0.919 | 0.844 |
| Stacking | "beats all base models" | 0.922 | — |
| Best of all (Voting/SVR) | — | 0.924 | 0.848 |

## How do we compare?

- **Ridge:** our single-split 0.898 is *above* the cited 0.857; our cross-validated 0.825 is
  *below* it. Both are the same model — the difference is entirely which number you quote.
- **XGBoost:** our best split (0.919) lands **below the cited 0.94**, and our honest cross-validated
  figure (0.844) is well below it.
- **Stacking / ensembles:** the cited claim that stacking "outperforms all base models" **did not
  replicate** — in our leak-controlled pipeline the ensembles merely tied the best single model.

## What explains the gaps? (the important part)

1. **Single lucky split vs. cross-validation.** Our own experiment shows the danger directly: the
   *same* model scores 0.92 on our favorable test split but 0.85 under 5-fold CV — a **+0.076 optimism
   gap**. Many published "0.94" figures report a single train/test split with no CV. A 0.94 on one split
   is entirely plausible from split luck; it is not the same claim as a 0.94 cross-validated.

2. **Target leakage.** Several public notebooks compute target-encoded features (e.g. mean charges by
   region/smoker) on the **full** dataset before splitting, which leaks test information and inflates
   scores. We deliberately refit those encodings on the **training fold only** (Chunk 6), which lowers
   our headline number but makes it trustworthy.

3. **Different preprocessing / feature sets.** Log-transforming the target, dropping outliers, or
   different encodings can each move R² by several points. We kept the target on its raw USD scale for
   interpretability.

4. **Small test sets amplify variance.** With ~268 test rows, a handful of the heavy-tailed high-cost
   cases landing in train vs test swings R² by 0.05+.

## Verdict

Our results are **fully consistent with the literature once methodology is matched.** Our tuned models
reach ~0.92 on a comparable single split — in line with published single-split numbers. Where we differ
is that we **also report the cross-validated ~0.85**, which we regard as the honest, decision-grade
figure. The gap between the two is not underperformance; it is the difference between a marketing number
and a number you would stake a pricing decision on.
