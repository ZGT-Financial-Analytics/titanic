from pathlib import Path
import pandas as pd

# paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
train_path = RAW / "train.csv"
test_path = RAW / "test.csv"

# read
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# shapes
print(f"train shape: {train.shape}  (expected (891, 12))")
print(f"test  shape: {test.shape}   (expected (418, 11))")

# peek
print("train.head():")
print(train.head())

# sanity checks: columns match Kaggle description
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
        # be helpful if they’re shuffled: show diffs
        missing = [c for c in expected if c not in got]
        extra = [c for c in got if c not in expected]
        raise AssertionError(
            f"{name} columns differ.\nExpected: {expected}\nGot     : {got}\n"
            f"Missing: {missing}\nExtra   : {extra}"
        )


assert_same_columns(train, expected_train, "train")
assert_same_columns(test, expected_test, "test")

print("Column checks passed.")
