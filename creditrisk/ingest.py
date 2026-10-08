"""Stage 1 - ingest + validate. Fails loudly on a broken schema, logs everything else."""
import json, logging, zipfile
from pathlib import Path
import pandas as pd
from . import config as C

log = logging.getLogger(__name__)

PAY = [f"PAY_{i}" for i in range(1, 7)]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT = [f"PAY_AMT{i}" for i in range(1, 7)]
REQUIRED = ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY + BILL + PAYAMT


def load_raw(path=None):
    path = Path(C.DATA_PATH if path is None else path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            df = pd.read_csv(z.open(name))
    else:
        df = pd.read_csv(path)
    df = df.rename(columns={"PAY_0": "PAY_1", C.TARGET_RAW: C.TARGET})
    log.info("Loaded %s rows x %s cols from %s", *df.shape, path.name)
    return df


def validate(df, need_target=True):
    """Return a data-quality report dict. Raises ValueError on critical problems."""
    missing_cols = [c for c in REQUIRED + ([C.TARGET] if need_target else []) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema check failed, missing columns: {missing_cols}")
    rep = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df[REQUIRED].isna().sum().sum()),
        "duplicate_ids": int(df["ID"].duplicated().sum()),
        "duplicate_rows": int(df.drop(columns=["ID"]).duplicated().sum()),
        "limit_nonpositive": int((df["LIMIT_BAL"] <= 0).sum()),
        "age_out_of_range": int((~df["AGE"].between(18, 100)).sum()),
        "pay_status_out_of_range": int((~df[PAY].isin(range(-2, 10)).all(axis=1)).sum()),
        "education_undocumented_codes_0_5_6": int(df["EDUCATION"].isin([0, 5, 6]).sum()),
        "marriage_undocumented_code_0": int((df["MARRIAGE"] == 0).sum()),
    }
    if need_target:
        if not set(df[C.TARGET].unique()) <= {0, 1}:
            raise ValueError("Target must be binary 0/1")
        rep["default_rate"] = round(float(df[C.TARGET].mean()), 4)
    if rep["missing_values"] > 0.05 * len(df):
        raise ValueError("More than 5% missing values - refusing to continue")
    log.info("Data-quality report: %s", json.dumps(rep))
    return rep
