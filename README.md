# Credit Default Model

A model predicting credit risk on the German Credit dataset.

`train.py` is the clean logistic regression baseline. `train_xgboost_mcts.py`
extends it with XGBoost and MCTS-based feature construction. Planned:
counterfactual stress testing.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py                 # logistic regression baseline
python train_xgboost_mcts.py    # XGBoost + MCTS-engineered features
```

The dataset is fetched automatically from OpenML the first time you run it, so
there is no file to download.

## What `train.py` does

1. Pulls the German Credit dataset from OpenML via its API.
2. Scales numeric features and one-hot encodes categorical ones.
3. Trains a logistic regression to predict bad credit risk.
4. Reports AUC and a full classification report.

## What `train_xgboost_mcts.py` does

1. Trains a baseline XGBoost model on the raw features.
2. Builds a pool of candidate engineered features — unary transforms
   (log1p, sqrt, square) and pairwise interactions (ratio, product) over the
   numeric columns (`mcts_features.build_candidates`).
3. Runs Monte Carlo Tree Search over subsets of that pool: each node is a
   subset of chosen candidates, children add one more candidate (with a
   higher index than the last one added, so every subset is reached via
   exactly one path), and the reward is cross-validated XGBoost AUC on the
   augmented feature set.
4. Retrains XGBoost on the original features plus the best subset MCTS
   found, and reports test AUC against the baseline.
