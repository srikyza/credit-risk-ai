"""Stage 3 - split, train 3 candidate models, tune the best family, select on validation."""
import logging
import numpy as np, pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from . import config as C

log = logging.getLogger(__name__)
CAT = ["EDUCATION"]


def split(df):
    tr_val, te = train_test_split(df, test_size=C.TEST_SIZE, stratify=df[C.TARGET], random_state=C.RANDOM_STATE)
    rel = C.VAL_SIZE / (1 - C.TEST_SIZE)
    tr, va = train_test_split(tr_val, test_size=rel, stratify=tr_val[C.TARGET], random_state=C.RANDOM_STATE)
    log.info("Split sizes  train=%d  val=%d  test=%d", len(tr), len(va), len(te))
    return tr, va, te


def _pipe(model, cols, scale):
    num = [c for c in cols if c not in CAT]
    pre = ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def train_and_select(tr, va, cols):
    X, y = tr[cols], tr[C.TARGET]
    cands = {}

    cands["Logistic Regression"] = _pipe(LogisticRegression(max_iter=2000, C=0.5), cols, True).fit(X, y)

    cands["Random Forest"] = _pipe(RandomForestClassifier(
        n_estimators=300, min_samples_leaf=30, max_features="sqrt", n_jobs=-1,
        random_state=C.RANDOM_STATE), cols, False).fit(X, y)

    search = RandomizedSearchCV(
        _pipe(HistGradientBoostingClassifier(max_iter=400, early_stopping=True,
                                             random_state=C.RANDOM_STATE), cols, False),
        {"model__learning_rate": loguniform(0.02, 0.2),
         "model__max_leaf_nodes": randint(8, 48),
         "model__min_samples_leaf": randint(20, 200),
         "model__l2_regularization": loguniform(1e-3, 10)},
        n_iter=12, cv=3, scoring="roc_auc", random_state=C.RANDOM_STATE, n_jobs=-1)
    search.fit(X, y)
    cands["Gradient Boosting (tuned)"] = search.best_estimator_
    log.info("Best GB params: %s", search.best_params_)

    rows = []
    for name, m in cands.items():
        p = m.predict_proba(va[cols])[:, 1]
        rows.append({"model": name, "val_roc_auc": roc_auc_score(va[C.TARGET], p),
                     "val_pr_auc": average_precision_score(va[C.TARGET], p)})
    board = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False).reset_index(drop=True)
    best = board.loc[0, "model"]
    log.info("Model leaderboard:\n%s", board.round(4).to_string(index=False))
    log.info("Selected model: %s", best)
    return cands[best], best, board
