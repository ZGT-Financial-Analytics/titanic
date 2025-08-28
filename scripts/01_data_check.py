# %% import statements
from __future__ import annotations

from titanic_lab.paths import (
    TRAIN_CSV,
    TEST_CSV,
)  # and whatever common files that were defined in the src/paths.py
import pandas as pd

# %%   verifying pandas version before running entire script
print(pd.__version__)


# %% file existence checks
print("TRAIN_CSV:", TRAIN_CSV.exists(), TRAIN_CSV)
print("TEST_CSV:", TEST_CSV.exists(), TEST_CSV)

# %% read CSV files
# pd.set_option("mode.dtype_backend", "pyarrow")  cannot use atm, buggy asf. Trouble shooting exhausted and failed.
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
# %% shape checks
print(f"train shape: {train.shape}  (expected (891, 12))")
print(f"test  shape: {test.shape}   (expected (418, 11))\n")
# %% head check
print("train.head():")
print(train.head(), "\n")
# %% schema check, expected
expected_train = [
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]
expected_test = [
    "PassengerId",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]


# %% column check function
def assert_same_columns(df, expected, name):
    got = list(df.columns)
    if got != expected:
        missing = [c for c in expected if c not in got]
        extra = [c for c in got if c not in expected]
        raise AssertionError(
            f"{name} columns differ.\nExpected: {expected}\nGot     : {got}\n"
            f"Missing: {missing}\nExtra   : {extra}"
        )


# %% run column checks
assert_same_columns(train, expected_train, "train")
assert_same_columns(test, expected_test, "test")
assert "Survived" not in test.columns, "Leakage: test set must not contain 'Survived'"
# %%    final checks print if all passed
print("Column checks passed.")
