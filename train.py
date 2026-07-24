"""
Baseline credit default model.

Fetches the German Credit dataset from OpenML (over its API), trains a
logistic regression to predict whether a loan is good or bad credit risk,
and prints AUC + a full classification report.

This is the clean baseline. Stress-testing / MCTS feature work comes later.
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report


def load_data():
    """Pull the German Credit dataset from OpenML via its API."""
    data = fetch_openml(name="credit-g", version=1, as_frame=True)
    X = data.data
    # Target is 'good' / 'bad'; map to 1 = bad credit risk (the event we care about)
    y = (data.target == "bad").astype(int)
    return X, y


def class_balance(y):
    """Print the class split (counts and percentages) for good vs. bad credit risk."""
    counts = y.value_counts().sort_index()
    labels = {0: "good", 1: "bad"}
    for cls, count in counts.items():
        print(f"{labels[cls]}: {count} ({count / len(y):.1%})")


def build_model(X):
    """Scale numeric columns, one-hot encode categoricals, then logistic regression."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def main():
    X, y = load_data()
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features.")
    class_balance(y)
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(X)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    print(f"AUC: {roc_auc_score(y_test, proba):.4f}\n")
    print(classification_report(y_test, pred, target_names=["good", "bad"], digits=2))


if __name__ == "__main__":
    main()
