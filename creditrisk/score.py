"""Stage 6 - batch / single scoring using the saved model artefact."""
import joblib, pandas as pd
from . import config as C
from .evaluate import assign_tier
from .features import prepare
from .ingest import REQUIRED


def load_artifact(path=None):
    return joblib.load(path or C.MODEL_PATH)


def score_frame(raw, artifact):
    """raw = DataFrame with the original 23 input columns (PAY_0 or PAY_1 both accepted)."""
    raw = raw.rename(columns={"PAY_0": "PAY_1"})
    if "ID" not in raw.columns:
        raw = raw.assign(ID=range(1, len(raw) + 1))
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    df, _ = prepare(raw)
    pdv = artifact["model"].predict_proba(df[artifact["columns"]])[:, 1]
    out = pd.DataFrame({"ID": df["ID"].values, "PROB_DEFAULT": pdv.round(4)})
    out["RISK_TIER"] = assign_tier(pdv).values
    out["DECISION"] = ["REVIEW / DECLINE" if v >= artifact["threshold"] else "APPROVE" for v in pdv]
    return out
