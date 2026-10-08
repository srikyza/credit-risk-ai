"""Stage 5 - drift monitoring with the Population Stability Index (PSI)."""
import logging
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config as C

log = logging.getLogger(__name__)


def psi(ref, new, bins=10):
    ref, new = np.asarray(ref, float), np.asarray(new, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                   # near-constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref); n = np.histogram(new, edges)[0] / len(new)
    r, n = np.clip(r, 1e-4, None), np.clip(n, 1e-4, None)
    return float(np.sum((n - r) * np.log(n / r)))


def status(v):
    return "ALERT" if v >= C.PSI_ALERT else "WATCH" if v >= C.PSI_WATCH else "stable"


def drift_report(ref_df, new_df, cols, ref_pd, new_pd):
    rows = [{"feature": c, "psi": psi(ref_df[c], new_df[c])} for c in cols]
    rows.append({"feature": "PREDICTED_PD (score)", "psi": psi(ref_pd, new_pd)})
    r = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    r["status"] = r["psi"].map(status)
    return r


def simulate_drift(df, severity=1.0, seed=C.RANDOM_STATE):
    """Create a 'bad economy' batch: bigger bills, more late payments, smaller repayments."""
    rng = np.random.default_rng(seed); d = df.copy()
    for c in [f"BILL_AMT{i}" for i in range(1, 7)]: d[c] = d[c] * (1 + 0.35 * severity)
    for c in [f"PAY_AMT{i}" for i in range(1, 7)]: d[c] = d[c] * (1 - 0.35 * severity)
    hit = rng.random(len(d)) < 0.25 * severity
    for c in [f"PAY_{i}" for i in range(1, 7)]: d.loc[hit, c] = np.minimum(d.loc[hit, c] + 1, 8)
    return d


def plot_drift(stable, drifted, fname="drift.png"):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    top = drifted.head(10)[::-1]; s = stable.set_index("feature")["psi"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y = np.arange(len(top))
    ax.barh(y + 0.2, top["psi"], 0.4, label="Simulated stress batch", color="#e53e3e")
    ax.barh(y - 0.2, s.reindex(top["feature"]).values, 0.4, label="Normal test batch", color="#2b6cb0")
    ax.axvline(C.PSI_WATCH, ls="--", c="orange"); ax.axvline(C.PSI_ALERT, ls="--", c="red")
    ax.set(yticks=y, yticklabels=top["feature"], xlabel="PSI", title="Drift monitor: normal vs stressed data")
    ax.legend(); fig.tight_layout(); fig.savefig(C.FIG_DIR / fname, dpi=130); plt.close(fig)
