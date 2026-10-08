"""Stage 4 - business-cost threshold, test metrics, plots, explainability, fairness audit."""
import logging
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from . import config as C

log = logging.getLogger(__name__)


def choose_threshold(y, p):
    """Pick the PD cut-off that minimises expected cost on the VALIDATION set."""
    grid = np.linspace(0.05, 0.95, 91)
    costs = []
    for t in grid:
        pred = p >= t
        fn = ((~pred) & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
        costs.append(C.COST_FN * fn + C.COST_FP * fp)
    return float(grid[int(np.argmin(costs))])


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def assign_tier(p):
    return pd.cut(pd.Series(p), C.TIER_BINS, labels=C.TIER_LABELS, include_lowest=True).astype(str)


def evaluate(model, cols, va, te, best_name):
    thr = choose_threshold(va[C.TARGET].values, model.predict_proba(va[cols])[:, 1])
    y = te[C.TARGET].values
    p = model.predict_proba(te[cols])[:, 1]
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    base_cost = C.COST_FN * y.sum()                       # approve everybody
    cost = C.COST_FN * fn + C.COST_FP * fp
    m = {
        "model": best_name, "threshold": round(thr, 3),
        "roc_auc": roc_auc_score(y, p), "gini": 2 * roc_auc_score(y, p) - 1,
        "ks": ks_stat(y, p), "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "expected_cost": float(cost), "cost_if_no_model": float(base_cost),
        "cost_saving_pct": float(100 * (1 - cost / base_cost)),
    }
    log.info("TEST  AUC=%.4f  KS=%.3f  recall=%.3f  precision=%.3f  cost saving=%.1f%%",
             m["roc_auc"], m["ks"], m["recall"], m["precision"], m["cost_saving_pct"])
    return m, p, thr


def plots(model, cols, te, p, thr, m):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    y = te[C.TARGET].values
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, lw=2, label=f"AUC = {m['roc_auc']:.3f}"); ax[0].plot([0, 1], [0, 1], "--", c="grey")
    ax[0].set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate"); ax[0].legend()
    pr, rc, _ = precision_recall_curve(y, p)
    ax[1].plot(rc, pr, lw=2, label=f"PR-AUC = {m['pr_auc']:.3f}"); ax[1].axhline(y.mean(), ls="--", c="grey", label="baseline")
    ax[1].set(title="Precision-recall (test)", xlabel="Recall", ylabel="Precision"); ax[1].legend()
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "roc_pr.png", dpi=130); plt.close(fig)

    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Pred good", "Pred default"], yticklabels=["Actual good", "Actual default"],
           title=f"Confusion matrix @ PD >= {thr:.2f}")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "confusion.png", dpi=130); plt.close(fig)

    s = te.sample(min(3000, len(te)), random_state=C.RANDOM_STATE)
    imp = permutation_importance(model, s[cols], s[C.TARGET], scoring="roc_auc", n_repeats=5,
                                 random_state=C.RANDOM_STATE, n_jobs=-1)
    fi = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)
    top = fi.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.barh(top.index, top.values, color="#2b6cb0")
    ax.set(title="What drives default risk? (permutation importance, AUC drop)", xlabel="Drop in AUC when shuffled")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "feature_importance.png", dpi=130); plt.close(fig)
    return fi


def tier_table(te, p):
    d = pd.DataFrame({"tier": assign_tier(p).values, "pd": p, "actual": te[C.TARGET].values})
    t = d.groupby("tier").agg(clients=("actual", "size"), avg_predicted_pd=("pd", "mean"),
                              actual_default_rate=("actual", "mean")).reindex(C.TIER_LABELS)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(t.index, t["actual_default_rate"] * 100, color=["#38a169", "#d69e2e", "#e53e3e"])
    for i, v in enumerate(t["actual_default_rate"] * 100): ax.text(i, v + 0.8, f"{v:.1f}%", ha="center")
    ax.set(title="Observed default rate by risk tier (test)", ylabel="% defaulted")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "risk_tiers.png", dpi=130); plt.close(fig)
    return t


def fairness_audit(te, p, thr):
    d = te.copy(); d["flag"] = (p >= thr).astype(int); d["pd"] = p
    out = []
    for col in C.AUDIT_COLUMNS:
        for g, sub in d.groupby(col):
            if len(sub) < 50: continue
            out.append({"attribute": col, "group": str(g), "n": len(sub),
                        "actual_default_rate": sub[C.TARGET].mean(),
                        "flagged_high_risk_rate": sub["flag"].mean(),
                        "avg_predicted_pd": sub["pd"].mean(),
                        "recall": ((sub["flag"] == 1) & (sub[C.TARGET] == 1)).sum() / max((sub[C.TARGET] == 1).sum(), 1)})
    return pd.DataFrame(out)
