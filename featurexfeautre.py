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
