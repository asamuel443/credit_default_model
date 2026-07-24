"""
XGBoost baseline vs. XGBoost + MCTS-engineered features, on the German
Credit dataset.

1. Trains a baseline XGBoost model on the raw features.
2. Runs MCTS over a pool of candidate engineered features (transforms and
   pairwise interactions of the numeric columns), searching for the subset
   that most improves cross-validated AUC.
3. Retrains XGBoost on the original features plus the chosen subset and
   compares test-set AUC against the baseline.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from train import load_data
from xgboost_model import build_xgboost_pipeline
from mcts_features import build_candidates, augment, search


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features.")
    print(f"Bad-credit rate: {y.mean():.1%}\n")

    # --- Baseline XGBoost, raw features ---
    baseline_model = build_xgboost_pipeline(X_train)
    baseline_model.fit(X_train, y_train)
    baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_proba)

    print("=== Baseline XGBoost (raw features) ===")
    print(f"Test AUC: {baseline_auc:.4f}\n")

    # --- MCTS feature search ---
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    candidates = build_candidates(X_train, numeric_cols)
    print(f"Searching over {len(candidates)} candidate engineered features via MCTS...\n")

    selected, cv_auc, baseline_cv_auc = search(X_train, y_train, candidates)
    selected_names = [candidates[i][0] for i in selected]

    print(f"Baseline CV AUC (no engineered features): {baseline_cv_auc:.4f}")
    print(f"Best CV AUC found:                         {cv_auc:.4f}")
    print(f"Selected features: {selected_names or '(none improved on baseline)'}\n")

    # --- Final XGBoost on augmented feature set ---
    X_train_aug = augment(X_train, candidates, selected)
    X_test_aug = augment(X_test, candidates, selected)

    final_model = build_xgboost_pipeline(X_train_aug)
    final_model.fit(X_train_aug, y_train)
    final_proba = final_model.predict_proba(X_test_aug)[:, 1]
    final_pred = final_model.predict(X_test_aug)
    final_auc = roc_auc_score(y_test, final_proba)

    print("=== XGBoost + MCTS-engineered features ===")
    print(f"Test AUC: {final_auc:.4f}\n")
    print(classification_report(y_test, final_pred, target_names=["good", "bad"], digits=2))

    print("=== Summary ===")
    print(f"Baseline test AUC:        {baseline_auc:.4f}")
    print(f"MCTS-augmented test AUC:  {final_auc:.4f}")
    print(f"Delta:                    {final_auc - baseline_auc:+.4f}")


if __name__ == "__main__":
    main()
