# titanic_wcg_xgb.py
import re
import numpy as np
import pandas as pd

# Fix: Add proper path imports
from titanic_lab.paths import TRAIN_CSV, TEST_CSV, ROOT

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# ---------- Feature helpers ----------
_TITLE_RE = re.compile(r",\s*([^\.]+)\.")


def normalize_title(raw):
    t = raw.strip().lower()
    # collapse rare titles; keep the ones we need for the WCG rule ("master")
    mp = {
        "ms": "miss",
        "mlle": "miss",
        "mme": "mrs",
        "lady": "rare",
        "the countess": "rare",
        "jonkheer": "rare",
        "sir": "rare",
        "don": "rare",
        "dona": "rare",
        "dr": "rare",
        "rev": "rare",
        "major": "rare",
        "col": "rare",
        "capt": "rare",
    }
    return mp.get(t, t)


def add_engineered_columns(df):
    out = df.copy()

    # Title from Name
    title = (
        out["Name"].str.extract(_TITLE_RE).iloc[:, 0].fillna("").map(normalize_title)
    )
    out["Title"] = title

    # Family size and an ID; a simple proxy for family groups
    surname = out["Name"].str.split(",").str[0].str.strip().str.lower()
    fam_size = (
        out["SibSp"].fillna(0).astype(int) + out["Parch"].fillna(0).astype(int) + 1
    )
    out["FamilySize"] = fam_size
    out["FamilyID"] = (surname + "_" + fam_size.astype(str)).astype("category")

    # DRY subgroup flags used everywhere
    low_title = out["Title"].str.lower()
    low_sex = out["Sex"].str.lower()
    out["IsBoy"] = low_title.eq("master")  # Title == "master"
    out["IsFemale"] = low_sex.eq("female")  # Sex == "female"

    return out


# ---------- The rules→then→model estimator ----------
class RuleThenXGB(BaseEstimator, ClassifierMixin):
    """
    First apply WCG-like hard rules (women/children-by-family),
    then send the uncovered rows to an XGBoost pipeline.
    """

    def __init__(self, base_model=None):
        # Store the parameter for sklearn compatibility
        self.base_model = base_model

        if base_model is None:
            self.base_model = XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=42,
                eval_metric="logloss",
            )

        # preprocess numeric + categorical for the XGB
        self.num_feats = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "Pclass"]
        self.cat_feats = ["Sex", "Embarked", "Title"]  # FamilyID not used by the model
        self.preprocess = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), self.num_feats),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    self.cat_feats,
                ),
            ],
            remainder="drop",
        )
        self.pipe = Pipeline([("prep", self.preprocess), ("xgb", self.base_model)])

        # learned from training split (FamilyID strings)
        self._families_where_all_females_or_boys_survived_ids = set()
        self._families_where_all_females_or_boys_died_ids = set()

    # --------- GATEWAY RULES (fit on training split ONLY) ----------
    # fb = "female OR boy('Master')". We precompute, per family,
    # whether ALL fb members lived (or died) in the training data.
    def _fit_family_rules(self, X, y):
        df = X.copy()
        y = pd.Series(y).astype(int)

        # Minimal table for grouping
        g = pd.DataFrame(
            {
                "fam": df["FamilyID"].astype(str),
                "fb": (df["IsBoy"] | df["IsFemale"]),
                "y": y.values,
            }
        )

        # Families that have at least one FB member
        has_fb = g.groupby("fam")["fb"].any().rename("has_fb")

        # Unanimity among FB members only (prefilter to FB rows)
        fb_rows = g[g["fb"]]
        unanim = fb_rows.groupby("fam")["y"].agg(
            all_fb_live=lambda s: s.eq(1).all(),
            all_fb_die=lambda s: s.eq(0).all(),
        )

        # Join and fill: families with no FB get False for both flags
        family_fb_summary = has_fb.to_frame().join(unanim, how="left").fillna(False)

        # Materialize the training-learned rule sets
        self._families_where_all_females_or_boys_survived_ids = set(
            family_fb_summary.index[family_fb_summary["all_fb_live"]].tolist()
        )
        self._families_where_all_females_or_boys_died_ids = set(
            family_fb_summary.index[family_fb_summary["all_fb_die"]].tolist()
        )

    # Masks for who is caught by the rules at predict time
    def _rule_masks(self, X):
        fam = X["FamilyID"].astype(str)
        is_boy = X["IsBoy"].values
        is_fem = X["IsFemale"].values

        live_mask = (
            is_boy
            & fam.isin(self._families_where_all_females_or_boys_survived_ids).values
        )  # predict 1
        die_mask = (
            is_fem & fam.isin(self._families_where_all_females_or_boys_died_ids).values
        )  # predict 0
        return live_mask, die_mask

    # --------- END GATEWAY RULES -----------------------------------

    def fit(self, X, y):
        X = add_engineered_columns(X)
        self._fit_family_rules(X, y)  # learn rule families from training split only
        live_m, die_m = self._rule_masks(X)
        covered = live_m | die_m

        # Train XGB ONLY on the uncovered rows (the "messy middle")
        self.pipe.fit(
            X.loc[~covered, self.num_feats + self.cat_feats], np.asarray(y)[~covered]
        )
        return self

    def predict_proba(self, X):
        X = add_engineered_columns(X)
        n = len(X)
        proba = np.empty((n, 2), dtype=float)
        proba[:] = np.nan

        live_m, die_m = self._rule_masks(X)
        proba[live_m] = [0.0, 1.0]
        proba[die_m] = [1.0, 0.0]

        need = np.isnan(proba).any(axis=1)
        if need.any():
            preds = self.pipe.predict_proba(
                X.loc[need, self.num_feats + self.cat_feats]
            )
            proba[need] = preds
        return proba

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------- Example usage ----------
if __name__ == "__main__":
    from sklearn.model_selection import cross_val_score

    # Load Titanic CSVs using proper path system
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    y = train["Survived"].astype(int)
    X = train.drop(columns=["Survived"])

    model = RuleThenXGB()

    # 🔍 EVALUATE WITH CROSS-VALIDATION FIRST
    print("🚀 EVALUATING HYBRID MODEL WITH 5-FOLD CV:")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"📊 CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print(f"📈 Individual folds: {[f'{score:.3f}' for score in cv_scores]}")

    if cv_scores.mean() < 0.84:
        print(
            f"⚠️  WARNING: CV accuracy ({cv_scores.mean():.1%}) is below promised 84%!"
        )
    else:
        print(f"✅ SUCCESS: CV accuracy ({cv_scores.mean():.1%}) meets/exceeds 84%!")

    # Now fit on all data and predict test
    print("\n🏗️  Training final model on all data...")
    model.fit(X, y)
    test_pred = model.predict(test)
    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})

    # Save to outputs directory
    output_path = ROOT / "outputs" / "submissions" / "submission_wcg_xgb.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
