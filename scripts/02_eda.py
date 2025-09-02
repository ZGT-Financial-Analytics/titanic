"""
Exploratory Data Analysis (EDA) script for the Titanic dataset.
This script performs the following steps:
1. Imports necessary modules and project paths.
2. Loads the Titanic training dataset.
3. Displays dataset shape and info.
4. Generates summary statistics for numeric and categorical features.
5. Checks for missing values in each column.
6. Counts non-missing values in the "Age" column.
7. Analyzes survival rates by various categories:
  - Sex
  - Passenger class (Pclass)
  - Embarkation port (Embarked)
  - Cross-tabulation of Sex and Pclass
8. Visualizes distributions of continuous features:
  - Age (histogram)
  - Fare (histogram, linear and log scale)
  - Boxplots of Age and Fare by survival status
9. Engineers and inspects categorical features:
  - Presence of Cabin information (HasCabin)
  - Family size (FamilySize) and binned family size (FamilyBin)
10. Saves key summary tables (pivot table of survival by Sex and Pclass, missingness per column) to the outputs directory.
Functions:
  plot_rate(series: pd.Series, title: str) -> None
    Plots a bar chart of survival rates for a given categorical grouping.
Outputs:
  - CSV files for pivot table and missingness statistics in the outputs/eda directory.
  - Multiple plots visualizing survival rates and feature distributions.
"""

# %% import statements
from __future__ import annotations
from titanic_lab.paths import (
    ROOT,
    TRAIN_CSV,
)  # and whatever common files that were defined in the src/paths.py
import pandas as pd
import matplotlib.pyplot as plt


# %% LOAD DATA
df = pd.read_csv(TRAIN_CSV)

# %% Final Shape Check
print("Shape:", df.shape)  # expect (891, 12)
print("\nInfo():")
df.info()

# %% generating summary statistics on dataset
num_desc = df.describe()
print(num_desc)

# %% generating summary stats on Categorical / strings / booleans
cat_desc = df.describe(include=["object", "category", "boolean"])
print(cat_desc)

# %% Checking missing values
na = df.isna().sum().sort_values(ascending=False)
print("\nMissing values per column:")
print(na)


# Using the df["Age"] column with the .notna() method chained to the sum method takes all of the values
# This counts the number of non-missing values in the "Age" column.
# .notna() returns a boolean Series (True for present values), and .sum() counts the Trues.

# %%
n_non_missing_age = df["Age"].notna().sum()
print(f"\nNon-missing Age count: {n_non_missing_age} (expect 714)")


# %% survival rate by category
def plot_rate(series: pd.Series, title: str) -> None:
    ax = series.sort_values(ascending=False).plot(kind="bar")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Survival rate")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# %%  binary target sanity check
print("\nSurvived value counts (0/1):")
print(df["Survived"].value_counts(dropna=False).sort_index())

# %% survival by Sex
rate_by_sex = df.groupby("Sex")["Survived"].mean()
plot_rate(rate_by_sex, "Survival rate by Sex")

# %% survival by Pclass
rate_by_pclass = df.groupby("Pclass")["Survived"].mean()
plot_rate(rate_by_pclass, "Survival rate by Pclass")

# %% survival by Embarked
rate_by_embarked = df.groupby("Embarked", dropna=False)["Survived"].mean()
plot_rate(rate_by_embarked, "Survival rate by Embarked")

# %% sex x Pclass (pivot to spot interactions)
pivot_sex_pclass = pd.pivot_table(
    df, index="Sex", columns="Pclass", values="Survived", aggfunc="mean"
)
print("\nPivot (Sex x Pclass) = mean survival:")
print(pivot_sex_pclass)

# %% plot grouped bars for Sex x Pclass
ax = pivot_sex_pclass.plot(kind="bar")
ax.set_ylim(0, 1)
ax.set_ylabel("Survival rate")
ax.set_title("Survival rate by Sex and Pclass")
plt.legend(title="Pclass")
plt.tight_layout()
plt.show()

# %% Age distribution
# Age distribution (note: missing values exist)
ax = df["Age"].dropna().plot(kind="hist", bins=30)
ax.set_title("Age distribution (train)")
ax.set_xlabel("Age")
plt.tight_layout()
plt.show()
# %% Fare distribution
# Fare is skewed; look at both linear and log scales
ax = df["Fare"].dropna().plot(kind="hist", bins=40)
ax.set_title("Fare distribution (linear scale)")
ax.set_xlabel("Fare")
plt.tight_layout()
plt.show()
# %% Fare distribution (log scale)
ax = df["Fare"].dropna().plot(kind="hist", bins=40, log=True)
ax.set_title("Fare distribution (log scale)")
ax.set_xlabel("Fare")
plt.tight_layout()
plt.show()
# %% Boxplots by survival
# Boxplots to compare distributions by survival
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df.boxplot(column="Age", by="Survived", ax=axes[0])
axes[0].set_title("Age by Survived")
axes[0].set_xlabel("Survived")
axes[0].set_ylabel("Age")
# %% Boxplot for Fare
df.boxplot(column="Fare", by="Survived", ax=axes[1])
axes[1].set_title("Fare by Survived")
axes[1].set_xlabel("Survived")
axes[1].set_ylabel("Fare")
fig.suptitle("")  # remove automatic suptitle
plt.tight_layout()
plt.show()

# ---------- 3.4 Handy categorical transforms ----------
# Example: small engineered flags that are useful to inspect
# %% Cabin presence
df["HasCabin"] = df["Cabin"].notna()
plot_rate(df.groupby("HasCabin")["Survived"].mean(), "Survival by Cabin presence")

# %% Family size (SibSp + Parch + self) binned
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["FamilyBin"] = pd.cut(
    df["FamilySize"], bins=[0, 1, 2, 4, 8, 20], labels=["1", "2", "3-4", "5-8", "9+"]
)
plot_rate(df.groupby("FamilyBin")["Survived"].mean(), "Survival by Family size")

# %%   persist tables
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
pivot_sex_pclass.to_csv(OUT / "pivot_sex_pclass.csv", index=True)
na.to_csv(OUT / "missingness.csv", header=["n_missing"])
print(f"\nSaved pivot and missingness tables to {OUT}")

# %%
