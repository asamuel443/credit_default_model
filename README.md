# Credit Default Model

A baseline model predicting credit risk on the German Credit dataset.

This is the clean starting point. Planned extensions: counterfactual stress
testing and MCTS-based feature construction.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py
```

The dataset is fetched automatically from OpenML the first time you run it, so
there is no file to download.

## What it does

1. Pulls the German Credit dataset from OpenML via its API.
2. Scales numeric features and one-hot encodes categorical ones.
3. Trains a logistic regression to predict bad credit risk.
4. Reports AUC and a full classification report.
