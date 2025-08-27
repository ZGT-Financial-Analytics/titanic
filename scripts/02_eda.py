from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Make `src/` importable without packaging (editable install is nicer, but this is lean)
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from paths import ROOT, TRAIN_CSV  # noqa: E402

# ---------- options ----------
# Arrow-backed dtypes: consistent nullable ints/strings + fast IO
pd.set_option("mode.dtype_backend", "pyarrow")

# ---------- load ----------
df = pd.read_csv(TRAIN_CSV, engine="pyarrow")

print("Shape:", df.shape)  # expect (891, 12)
print("\nInfo():")
print(df.info())

# ---------- 3.1 Summary statistics ----------
# Numeric
num_desc = df.describe(numeric_only=True)

# Categorical / strings / booleans (Arrow- or pandas-backed)
cat_desc = df.describe(include=["string", "category", "boolean"])


# Missingness overview
na = df.isna().sum().sort_values(ascending=False)
print("\nMissing values per column:")
print(na)

# Quick checks that match the competition description
n_non_missing_age = df["Age"].notna().sum()
print(f"\nNon-missing Age count: {n_non_missing_age} (expect 714)")


# ---------- 3.2 Survival rates by category ----------
def plot_rate(series: pd.Series, title: str) -> None:
    ax = series.sort_values(ascending=False).plot(kind="bar")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Survival rate")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# Binary target sanity
print("\nSurvived value counts (0/1):")
print(df["Survived"].value_counts(dropna=False).sort_index())

# Survival by Sex
rate_by_sex = df.groupby("Sex")["Survived"].mean()
plot_rate(rate_by_sex, "Survival rate by Sex")

# Survival by Pclass
rate_by_pclass = df.groupby("Pclass")["Survived"].mean()
plot_rate(rate_by_pclass, "Survival rate by Pclass")

# Survival by Embarked (includes NaN bucket)
rate_by_embarked = df.groupby("Embarked", dropna=False)["Survived"].mean()
plot_rate(rate_by_embarked, "Survival rate by Embarked")

# Cross-tab: Sex x Pclass (pivot to spot interactions)
pivot_sex_pclass = pd.pivot_table(
    df, index="Sex", columns="Pclass", values="Survived", aggfunc="mean"
)
print("\nPivot (Sex x Pclass) = mean survival:")
print(pivot_sex_pclass)

# Plot grouped bars for Sex x Pclass
ax = pivot_sex_pclass.plot(kind="bar")
ax.set_ylim(0, 1)
ax.set_ylabel("Survival rate")
ax.set_title("Survival rate by Sex and Pclass")
plt.legend(title="Pclass")
plt.tight_layout()
plt.show()

# ---------- 3.3 Distributions of continuous features ----------
# Age distribution (note: missing values exist)
ax = df["Age"].dropna().plot(kind="hist", bins=30)
ax.set_title("Age distribution (train)")
ax.set_xlabel("Age")
plt.tight_layout()
plt.show()

# Fare is skewed; look at both linear and log scales
ax = df["Fare"].dropna().plot(kind="hist", bins=40)
ax.set_title("Fare distribution (linear scale)")
ax.set_xlabel("Fare")
plt.tight_layout()
plt.show()

ax = df["Fare"].dropna().plot(kind="hist", bins=40, log=True)
ax.set_title("Fare distribution (log scale)")
ax.set_xlabel("Fare")
plt.tight_layout()
plt.show()

# Boxplots to compare distributions by survival
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df.boxplot(column="Age", by="Survived", ax=axes[0])
axes[0].set_title("Age by Survived")
axes[0].set_xlabel("Survived")
axes[0].set_ylabel("Age")

df.boxplot(column="Fare", by="Survived", ax=axes[1])
axes[1].set_title("Fare by Survived")
axes[1].set_xlabel("Survived")
axes[1].set_ylabel("Fare")
fig.suptitle("")  # remove automatic suptitle
plt.tight_layout()
plt.show()

# ---------- 3.4 Handy categorical transforms ----------
# Example: small engineered flags that are useful to inspect
df["HasCabin"] = df["Cabin"].notna()
plot_rate(df.groupby("HasCabin")["Survived"].mean(), "Survival by Cabin presence")

# Family size (SibSp + Parch + self) binned
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["FamilyBin"] = pd.cut(
    df["FamilySize"], bins=[0, 1, 2, 4, 8, 20], labels=["1", "2", "3-4", "5-8", "9+"]
)
plot_rate(df.groupby("FamilyBin")["Survived"].mean(), "Survival by Family size")

# ---------- persist a couple of tables ----------
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
pivot_sex_pclass.to_csv(OUT / "pivot_sex_pclass.csv", index=True)
na.to_csv(OUT / "missingness.csv", header=["n_missing"])
print(f"\nSaved pivot and missingness tables to {OUT}")
