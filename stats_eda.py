# %%
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import json
from datetime import datetime

# creating mean difference test function

try:
    from titanic_lab.paths import ROOT, TRAIN_CSV  # type: ignore[attr-defined]

    DATA_ROOT = Path(ROOT)
    TRAIN_PATH = Path(TRAIN_CSV)
except Exception:
    DATA_ROOT = Path(".")
    TRAIN_PATH = DATA_ROOT / "data" / "train.csv"
df = pd.read_csv(TRAIN_PATH)
# %%
# title
df["Title"] = (
    df["Name"].str.extract(r",\s*([^\.]+)\.").iloc[:, 0].str.strip().str.lower()
)

# boys (<18, male, age present)
boys = df[df["Sex"].str.lower().eq("male") & df["Age"].notna() & df["Age"].lt(18)]

# survival rates
rate_master = boys[boys["Title"].eq("master")]["Survived"].mean()
rate_nonmaster = boys[~boys["Title"].eq("master")]["Survived"].mean()

# bar plot
labels = ["Master", "Non-Master"]
rates = [rate_master, rate_nonmaster]

plt.figure(figsize=(5, 4))
plt.bar(labels, rates)
plt.ylim(0, 1)
plt.ylabel("Survival rate")
plt.title("Boys (<18, male): Survival by 'Master' title")
for i, v in enumerate(rates):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.show()

# %%
print(df.columns)
# %%
# counts
n_master = boys["Title"].eq("master").sum()
n_nonmaster = (~boys["Title"].eq("master")).sum()
counts = [int(n_master), int(n_nonmaster)]

# update labels to show counts
labels = [f"Master (n={counts[0]})", f"Non-Master (n={counts[1]})"]

# annotate bars with rate + count
plt.figure(figsize=(5, 4))
plt.bar(labels, rates)
plt.ylim(0, 1)
plt.ylabel("Survival rate")
plt.title("Boys (<18, male): Survival by 'Master' title")
for i, v in enumerate(rates):
    plt.text(i, v + 0.02, f"{v:.2f}\n(n={counts[i]})", ha="center", va="bottom")
plt.tight_layout()
plt.show()
# %%
# %%
# SED and t-score (Welch) for survival-rate means: Master vs Non-Master (boys < 18)
g1 = boys.loc[boys["Title"].eq("master"), "Survived"].astype(float)
g2 = boys.loc[~boys["Title"].eq("master"), "Survived"].astype(float)

n1, n2 = g1.size, g2.size
m1, m2 = g1.mean(), g2.mean()
s1, s2 = g1.var(ddof=1), g2.var(ddof=1)  # sample variances

sed = np.sqrt(s1 / n1 + s2 / n2)  # standard error of the difference in means (Welch)
t_score = (m1 - m2) / sed

# Welch-Satterthwaite df (optional to print, used here for completeness)
df_welch = (s1 / n1 + s2 / n2) ** 2 / (
    (s1**2) / ((n1**2) * (n1 - 1)) + (s2**2) / ((n2**2) * (n2 - 1))
)
p_two_sided = 2 * stats.t.sf(np.abs(t_score), df=df_welch)

print(f"Master vs Non-Master (boys <18):")
print(f"n1={n1}, n2={n2}")
print(f"mean1={m1:.4f}, mean2={m2:.4f}, diff={m1 - m2:.4f}")
print(f"SED={sed:.6f}")
print(f"t-score={t_score:.4f}  (df≈{df_welch:.1f}, p={p_two_sided:.4g})")
# %%


OUT_METRICS = Path("outputs/metrics")
OUT_METRICS.mkdir(parents=True, exist_ok=True)
out_path = OUT_METRICS / "survival_master_vs_nonmaster.json"

results = {
    "comparison": "boys(<18): Master vs Non-Master",
    "n1_master": int(n1),
    "n2_nonmaster": int(n2),
    "mean1_master": round(m1, 1),
    "mean2_nonmaster": round(m2, 1),
    "diff_means": round(m1 - m2, 1),
    "SED": round(sed, 1),
    "t_score": round(t_score, 1),
    "timestamp": datetime.now().isoformat(timespec="seconds"),
}


with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1)

# %%
