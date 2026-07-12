# Stage 4 · Chunk 34 — Limitations & Future Work

## Limitations

### Data limitations
- **Small, cross-sectional sample.** 1,337 rows, one snapshot per person. No longitudinal view of how
  an individual's costs evolve, and the heavy-tailed target means ~268 test rows give noisy point
  estimates (the ±0.04 CV std we measured).
- **Only 7 raw features.** The dataset has no diagnoses, claims history, chronic conditions, medications,
  income, occupation, or lifestyle detail. Our residual analysis showed the model's biggest misses are
  high-cost **non-smokers** whose drivers are simply absent from the data — an irreducible ceiling with
  these columns.
- **Unknown provenance.** Collection method and vintage are undocumented; `smoker` is likely
  self-reported (a known under-reporting bias), and the USD cost scale may not reflect current prices.
- **Class imbalance.** Only 20.5% smokers; sub-segment estimates (e.g. young normal-BMI smokers) rest on
  few observations.

### Model limitations
- **~0.85 cross-validated R² is a real ceiling** for this feature set — ~15% of cost variation is
  unexplained, concentrated in the high-cost tail.
- **Residuals are non-normal** (heavy right tail), so naive Gaussian prediction intervals would be wrong;
  quantile or conformal intervals are needed for uncertainty quantification.
- **Feature redundancy** (multiple smoking encodings) makes linear coefficients uninterpretable — we
  relied on SHAP instead, which is robust to this.

### Business / generalizability limitations
- Findings are specific to this population and cost regime; they may not transfer to other geographies,
  payer systems, or time periods.
- Correlation, not proven causation: cessation/weight programs are *expected* to lower cost based on the
  associations here, but the dataset cannot prove the counterfactual.

## Future work

### More data (highest value)
- **Claims & diagnosis history, chronic-condition flags** — directly targets the model's blind spot
  (unexplained high-cost non-smokers) and would lift the ~0.85 ceiling.
- **Longitudinal panels** to model cost trajectories and program effects over time.

### Modeling
- **Tweedie / Gamma GLMs** — purpose-built for right-skewed, non-negative insurance cost targets.
- **Quantile regression / conformal prediction** — honest prediction intervals for pricing.
- **Log-target modeling** and monotonic constraints (e.g. cost non-decreasing in age) for defensibility.
- **Deep learning** only if the feature set grows substantially — not warranted for 7 features.

### Process & scale
- Wrap the pipeline in a reproducible package with automated retraining, drift monitoring, and fairness
  audits (with sex/region explicitly excluded as rating factors).

### Open questions
- Why do specific non-smokers incur very high costs? (needs clinical data)
- What is the *causal* claims impact of cessation/weight programs? (needs an intervention study)
- How stable are these drivers over time and across regions?
