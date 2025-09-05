# Titanic (scikit-learn) — VS Code starter

Minimal project using Pandas/NumPy/scikit-learn with Kaggle Titanic data.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Get data

Option A: From Kaggle website, download `train.csv` and `test.csv` and place them into `data/raw/`.

Option B: Kaggle CLI:

```bash
pip install kaggle
# Put kaggle.json (API token) in ~/.kaggle/kaggle.json with 600 perms, then:
kaggle competitions download -c titanic -p data/raw
unzip data/raw/titanic.zip -d data/raw
```

## Run

- VS Code: Terminal → Run Task → `train`, then `predict`.


`submission.csv` is ready to upload to Kaggle.

```

## Notes
- Features are in `src/features.py`. Adjust as needed.
- Model: `Pipeline(preprocess → LogisticRegression)`. Edit in `src/train.py`.
