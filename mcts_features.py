"""
MCTS-based feature construction.

Searches over a pool of candidate engineered features (unary transforms and
pairwise interactions of the numeric columns) for the subset that most
improves cross-validated XGBoost AUC. Each MCTS node is a subset of chosen
candidates; children only add candidates with a higher index than the last
one added, so every reachable subset corresponds to exactly one path (no
permutation duplicates).
"""

import math
import random

import numpy as np
from sklearn.model_selection import cross_val_score

from xgboost_model import build_xgboost_pipeline


def build_candidates(X, numeric_cols):
    """Return a list of (name, fn) pairs. Each fn(X) -> pd.Series."""
    candidates = []

    for col in numeric_cols:
        col_min = X[col].min()
        candidates.append((f"{col}_log1p", lambda X, c=col, m=col_min: np.log1p(X[c] - m + 1)))
        candidates.append((f"{col}_sqrt", lambda X, c=col: np.sqrt(X[c].abs())))
        candidates.append((f"{col}_sq", lambda X, c=col: X[c] ** 2))

    for a, b in [(a, b) for i, a in enumerate(numeric_cols) for b in numeric_cols[i + 1:]]:
        candidates.append((f"{a}_over_{b}", lambda X, a=a, b=b: X[a] / (X[b] + 1e-6)))
        candidates.append((f"{a}_times_{b}", lambda X, a=a, b=b: X[a] * X[b]))

    return candidates


def augment(X, candidates, selected):
    X_aug = X.copy()
    for i in selected:
        name, fn = candidates[i]
        X_aug[name] = fn(X)
    return X_aug


class MCTSNode:
    def __init__(self, selected, last_index, parent=None):
        self.selected = selected
        self.last_index = last_index
        self.parent = parent
        self.children = {}
        self.untried = None
        self.visits = 0
        self.total_reward = 0.0

    @property
    def value(self):
        return self.total_reward / self.visits if self.visits else 0.0


def _ucb1(child, parent_visits, exploration):
    if child.visits == 0:
        return float("inf")
    return child.value + exploration * math.sqrt(math.log(parent_visits) / child.visits)


def search(X_train, y_train, candidates, iterations=40, max_depth=4, cv_folds=3,
           exploration=0.05, seed=42):
    """Run MCTS over candidate feature subsets. Returns (best_subset, best_cv_auc, baseline_cv_auc)."""
    rng = random.Random(seed)
    root = MCTSNode(selected=(), last_index=-1)
    cache = {}

    def evaluate(selected):
        if selected not in cache:
            X_aug = augment(X_train, candidates, selected)
            model = build_xgboost_pipeline(X_aug)
            cache[selected] = cross_val_score(
                model, X_aug, y_train, cv=cv_folds, scoring="roc_auc"
            ).mean()
        return cache[selected]

    baseline_auc = evaluate(())

    for _ in range(iterations):
        node = root

        # selection: descend via UCB1 until a node has untried actions or is terminal
        while True:
            if node.untried is None:
                node.untried = [i for i in range(len(candidates)) if i > node.last_index]
                rng.shuffle(node.untried)
            at_max_depth = len(node.selected) >= max_depth
            if node.untried or at_max_depth or not node.children:
                break
            node = max(node.children.values(), key=lambda c: _ucb1(c, node.visits, exploration))

        # expansion
        if len(node.selected) < max_depth and node.untried:
            action = node.untried.pop()
            new_selected = tuple(sorted(node.selected + (action,)))
            child = MCTSNode(new_selected, last_index=action, parent=node)
            node.children[action] = child
            node = child

        # evaluation (direct — no random rollout, since evaluating IS the expensive step)
        reward = evaluate(node.selected)

        # backpropagation
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.total_reward += reward
            cur = cur.parent

    best = root
    stack = [root]
    while stack:
        n = stack.pop()
        if n.visits and n.value > best.value:
            best = n
        stack.extend(n.children.values())

    return best.selected, best.value, baseline_auc
