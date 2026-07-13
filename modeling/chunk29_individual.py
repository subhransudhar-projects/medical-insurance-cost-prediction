"""
Stage 4 · Chunk 29 — Individual prediction explanations (SHAP waterfalls).

Five representative customers: we show actual vs predicted charges and a
waterfall of how the model built each prediction from the base value.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from model_utils import load_splits, load_model, set_style, save_fig, VERMILLION, BLUE, NAVY

set_style()
X_train, X_test, y_train, y_test = load_splits(scaled=False)
gb = load_model("gradient_boosting")
explainer = shap.TreeExplainer(gb)
base = float(np.ravel(explainer.expected_value)[0])

age, bmi, smk, ch = X_test["age"], X_test["bmi"], X_test["smoker"], X_test["children"]
profiles = [
    ("Young non-smoker,\nnormal BMI", (age <= 30) & (smk == 0) & (bmi.between(18.5, 25))),
    ("Older smoker,\nhigh BMI",       (age >= 50) & (smk == 1) & (bmi >= 30)),
    ("Young smoker,\nnormal BMI",      (age <= 32) & (smk == 1) & (bmi.between(18.5, 27))),
    ("Older non-smoker,\nhigh BMI",    (age >= 50) & (smk == 0) & (bmi >= 30)),
    ("Middle-aged,\n3+ children",      (age.between(35, 50)) & (ch >= 3)),
]

# pick first matching test row per profile
idxs, used = [], set()
for label, mask in profiles:
    cand = [i for i in X_test.index[mask] if i not in used]
    if not cand:
        cand = [i for i in X_test.index if i not in used]  # fallback
    idxs.append(cand[0]); used.add(cand[0])

X_sel = X_test.loc[idxs]
sv = explainer.shap_values(X_sel)                       # (5, n_features)
preds = gb.predict(X_sel)
actuals = y_test.loc[idxs].values

def waterfall(ax, contribs, predicted, actual, title):
    s = contribs.reindex(contribs.abs().sort_values(ascending=False).index)
    top = s.head(6)
    other = s.iloc[6:].sum()
    items = list(top.items())
    if abs(other) > 1:
        items.append(("other features", other))
    # cumulative bar start positions + x-span (for label padding)
    starts, running, xs = [], base, [base]
    for _, v in items:
        starts.append(running); running += v; xs.append(running)
    pad = (max(xs) - min(xs)) * 0.02 or 100.0
    for i, ((k, v), start) in enumerate(zip(items, starts)):
        end = start + v
        ax.barh(i, v, left=start, color=(VERMILLION if v > 0 else BLUE), edgecolor="white")
        # label placed just OUTSIDE the bar end in dark text -> always readable
        if v >= 0:
            ax.text(end + pad, i, f"+${v:,.0f}", ha="left", va="center",
                    fontsize=8, color=NAVY, fontweight="semibold")
        else:
            ax.text(end - pad, i, f"-${abs(v):,.0f}", ha="right", va="center",
                    fontsize=8, color=NAVY, fontweight="semibold")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([k for k, _ in items], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(base, color="grey", linestyle=":", linewidth=1.2)
    ax.axvline(running, color="black", linewidth=1.4)
    ax.set_title(f"{title}\nactual \\${actual:,.0f}  |  predicted \\${predicted:,.0f}",
                 loc="left", fontsize=10.5)
    ax.set_xlabel("charges (USD)", fontsize=9)
    # asymmetric x-limits: extra room on the left so left-ending bar labels
    # sit in open space rather than colliding with the y-axis feature names
    lo, hi = min(xs), max(xs)
    span = (hi - lo) or 1.0
    ax.set_xlim(lo - 0.30 * span, hi + 0.16 * span)

print("=" * 70); print("CHUNK 29 — INDIVIDUAL PREDICTION EXPLANATIONS"); print("=" * 70)
print(f"Model base value (avg prediction): ${base:,.0f}\n")

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.ravel()
for j, (label, _) in enumerate(profiles):
    row = X_sel.iloc[j]
    contribs = pd.Series(sv[j], index=X_sel.columns)
    waterfall(axes[j], contribs, preds[j], actuals[j], f"({chr(97+j)}) {label}")
    print(f"CASE {j+1}: {label.replace(chr(10),' ')}")
    print(f"  profile: age={int(row['age'])}, bmi={row['bmi']:.1f}, "
          f"smoker={'yes' if row['smoker']==1 else 'no'}, children={int(row['children'])}")
    print(f"  actual ${actuals[j]:,.0f} | predicted ${preds[j]:,.0f} | error ${actuals[j]-preds[j]:,.0f}")
    top3 = contribs.reindex(contribs.abs().sort_values(ascending=False).index).head(3)
    print(f"  top drivers: " + "; ".join(f"{k} {'+' if v>=0 else '-'}${abs(v):,.0f}" for k, v in top3.items()))
    print()

axes[5].set_visible(False)
fig.suptitle("SHAP Waterfalls — How the Model Builds Each Prediction (base = $%s)" % f"{base:,.0f}",
             fontsize=15, fontweight="bold", y=1.0)
fig.tight_layout()
save_fig(fig, "chunk29_individual_waterfalls")
