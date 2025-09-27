from pathlib import Path
import joblib  # noqa: F401
import pandas as pd
import numpy as np  # noqa: F401
from titanic_lab.paths import ROOT, TRAIN_CSV

try:
    from titanic_lab.paths import TEST_CSV  # type: ignore
except Exception:
    TEST_CSV = Path(ROOT) / "data" / "test.csv"
# storage for outputs
OUT_MODELS = Path(ROOT) / "outputs" / "models"
OUT_SUB = Path(ROOT) / "outputs" / "submissions"
OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_SUB.mkdir(parents=True, exist_ok=True)


def predict_gender_rule(df):
    """Return Survived (positive class) if female, return Not Survived (negative class) if male"""
    return np.where(df["Sex"] == "female", 1, 0)


def main() -> None:
    # Load data
    pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    pred_test = predict_gender_rule(test)

    # Build submission
    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            "Survived": pred_test,
        }
    )

    out_path = OUT_SUB / "submission_gender_rule.csv"
    submission.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
