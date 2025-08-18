import re
import pandas as pd


# Simple, Kaggle-proven features for Titanic
def extract_title(name: str) -> str:
    # e.g., "Braund, Mr. Owen Harris" -> "Mr"
    m = re.search(r",\s*([^\.]+)\.", name)
    return m.group(1).strip() if m else "Unknown"


def normalize_title(title: str) -> str:
    title = title.strip()
    common = {
        "Mr",
        "Mrs",
        "Miss",
        "Master",
        "Dr",
        "Rev",
        "Col",
        "Mlle",
        "Mme",
        "Ms",
        "Major",
        "Lady",
        "Sir",
        "Don",
        "Jonkheer",
        "Capt",
        "Countess",
        "Dona",
    }
    if title in {"Mlle"}:
        return "Miss"  # noqa: E701
    if title in {"Ms"}:
        return "Miss"
    if title in {"Mme"}:
        return "Mrs"
    if title in {"Lady", "Countess"}:
        return "Noble"
    if title in {"Sir", "Don", "Dona", "Jonkheer"}:
        return "Noble"
    if title in {"Col", "Major", "Capt"}:
        return "Officer"
    return title if title in common else "Rare"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Derived columns
    out["Title"] = out["Name"].map(extract_title).map(normalize_title)
    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["CabinKnown"] = out["Cabin"].notna().astype(int)
    out["TicketPrefix"] = (
        out["Ticket"]
        .astype(str)
        .str.replace(r"\d", "", regex=True)
        .str.strip()
        .replace("", "NONE")
    )

    # Drop high-cardinality columns you don't want to one-hot fully
    drop_cols = ["Name", "Cabin", "Ticket"]
    keep = [c for c in out.columns if c not in drop_cols]
    return out[keep]


def split_target(df: pd.DataFrame, target="Survived"):
    y = df[target] if target in df.columns else None
    X = df.drop(columns=[target]) if target in df.columns else df
    return X, y
