import argparse
import joblib
import pandas as pd
from features import build_features, split_target

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv", type=str, required=True, help="data/raw/test.csv from Kaggle Titanic")
    p.add_argument("--model", type=str, required=True, help="models/titanic.joblib")
    p.add_argument("--out_csv", type=str, default="submission.csv")
    return p.parse_args()

def main():
    args = parse_args()
    bundle = joblib.load(args.model)
    model = bundle["model"]

    test_df = pd.read_csv(args.test_csv)
    test_feat = build_features(test_df)
    X, _ = split_target(test_feat, target="Survived")  # Survived not in test

    preds = model.predict(X)
    sub = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": preds.astype(int)
    })
    sub.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(sub)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
