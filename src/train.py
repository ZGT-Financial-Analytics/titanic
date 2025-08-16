import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import build_features, split_target

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True, help="data/raw/train.csv from Kaggle Titanic")
    p.add_argument("--model_out", type=str, default="models/titanic.joblib")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cv", type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args()

    df = pd.read_csv(args.train_csv)
    df_feat = build_features(df)
    X, y = split_target(df_feat, target="Survived")

    # Identify column groups
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols),
        ]
    )

    clf = LogisticRegression(max_iter=1000, n_jobs=None)

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    scores = cross_validate(
        pipe, X, y,
        scoring={"acc":"accuracy","f1":"f1"},
        cv=cv, n_jobs=None, return_train_score=False
    )
    print(f"CV accuracy: {scores['test_acc'].mean():.4f} ± {scores['test_acc'].std():.4f}")
    print(f"CV F1:       {scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}")

    pipe.fit(X, y)
    joblib.dump({"model": pipe, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols}, args.model_out)
    print(f"Saved model to {args.model_out}")

if __name__ == "__main__":
    main()
