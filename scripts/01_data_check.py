from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

# import shared paths from src/
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from paths import TRAIN_CSV, TEST_CSV  # noqa: E402

# Check that the data files exist
# %%
print("TRAIN_CSV:", TRAIN_CSV.exists(), TRAIN_CSV)
print("TEST_CSV:", TEST_CSV.exists(), TEST_CSV)

pd.set_option("mode.dtype_backend", "pyarrow")

# --- load both files ---
train = pd.read_csv(TRAIN_CSV, engine="pyarrow")
test = pd.read_csv(TEST_CSV, engine="pyarrow")

# --- shape checks (canonical Titanic sizes) ---
print(f"train shape: {train.shape}  (expected (891, 12))")
print(f"test  shape: {test.shape}   (expected (418, 11))\n")

# Quick peek to verify we're looking at Titanic data
print("train.head():")
print(train.head(), "\n")

# --- schema checks: exact columns & order ---
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


def assert_same_columns(df, expected, name):
    got = list(df.columns)
    if got != expected:
        missing = [c for c in expected if c not in got]
        extra = [c for c in got if c not in expected]
        raise AssertionError(
            f"{name} columns differ.\nExpected: {expected}\nGot     : {got}\n"
            f"Missing: {missing}\nExtra   : {extra}"
        )


assert_same_columns(train, expected_train, "train")
assert_same_columns(test, expected_test, "test")
assert "Survived" not in test.columns, "Leakage: test set must not contain 'Survived'"

print("Column checks passed.")

# %%
