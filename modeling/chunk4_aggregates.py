"""
Stage 4 · Chunk 4 — Feature engineering: aggregated (group-level) features.

Adds group context:
  - region_mean_charges : mean charges within each region   (TARGET-derived)
  - smoker_mean_charges : mean charges by smoker status      (TARGET-derived)
  - region_mean_bmi     : mean bmi within each region        (feature-derived, safe)

IMPORTANT — data-leakage note:
The two *_mean_charges columns are TARGET ENCODINGS (built from `charges`, the
very thing we predict). Computed on the FULL dataset they leak test information
into training and inflate scores. Here we compute them on full data ONLY to
reveal the pattern for the story; Chunk 6 REFITS them on the training split
alone (test rows mapped through the train-fitted table) so the reported metrics
stay honest. region_mean_bmi is derived from a feature, not the target, so it
carries no target leakage.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
FEATURES = BASE / "insurance_features.csv"

df = pd.read_csv(FEATURES)

region_charges_map = df.groupby("region")["charges"].mean()
smoker_charges_map = df.groupby("smoker")["charges"].mean()
region_bmi_map     = df.groupby("region")["bmi"].mean()

df["region_mean_charges"] = df["region"].map(region_charges_map)
df["smoker_mean_charges"] = df["smoker"].map(smoker_charges_map)
df["region_mean_bmi"]     = df["region"].map(region_bmi_map)

print("=" * 68)
print("CHUNK 4 — AGGREGATED FEATURES")
print("=" * 68)
print(f"\nShape now : {df.shape[0]} rows x {df.shape[1]} columns")

print("\n--- Mean charges by region (region_mean_charges) ---")
print(region_charges_map.round(0).sort_values(ascending=False).to_string())

print("\n--- Mean charges by smoker status (smoker_mean_charges) ---")
print(smoker_charges_map.round(0).to_string())

print("\n--- Mean BMI by region (region_mean_bmi) ---")
print(region_bmi_map.round(2).sort_values(ascending=False).to_string())

print("\n--- First 5 rows (new columns) ---")
show = ["region", "region_mean_charges", "region_mean_bmi", "smoker", "smoker_mean_charges"]
print(df[show].head(5).to_string(index=False))

print("\n" + "!" * 68)
print("LEAKAGE CAVEAT: region_mean_charges & smoker_mean_charges are TARGET")
print("encodings. Full-data values shown here are for insight only; Chunk 6")
print("recomputes them from the TRAINING split alone before any scoring.")
print("Also note: smoker_mean_charges takes just 2 values (one per smoker class)")
print("— it is essentially a target-scaled version of the smoker flag.")
print("!" * 68)

df.to_csv(FEATURES, index=False)
print(f"\n[saved] {FEATURES.name}  ({df.shape[1]} columns)")
