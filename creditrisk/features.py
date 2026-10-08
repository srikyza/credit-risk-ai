"""Stage 2 - cleaning + feature engineering. Stateless (no fitting) so it can never leak."""
import numpy as np, pandas as pd
from . import config as C
from .ingest import PAY, BILL, PAYAMT


def clean(df):
    df = df.copy()
    log = {}
    m = df["EDUCATION"].isin([0, 5, 6]); log["EDUCATION 0/5/6 -> 4 (other)"] = int(m.sum())
    df.loc[m, "EDUCATION"] = 4
    m = df["MARRIAGE"] == 0; log["MARRIAGE 0 -> 3 (other)"] = int(m.sum())
    df.loc[m, "MARRIAGE"] = 3
    df["AGE_BAND"] = pd.cut(df["AGE"], [0, 29, 39, 49, 120], labels=["<30", "30-39", "40-49", "50+"]).astype(str)
    return df, log


def engineer(df):
    df = df.copy()
    lim = df["LIMIT_BAL"].replace(0, np.nan)
    util = (df[BILL].div(lim, axis=0)).clip(-1, 5)
    df["UTIL_LATEST"] = util["BILL_AMT1"]
    df["UTIL_MEAN"] = util.mean(axis=1)
    df["UTIL_MAX"] = util.max(axis=1)
    df["BILL_TREND"] = ((df["BILL_AMT1"] - df["BILL_AMT6"]) / lim).clip(-5, 5)
    # a payment in month k settles the bill of month k+1 (older month)
    paid = df[PAYAMT[:5]].sum(axis=1)
    owed = df[BILL[1:]].clip(lower=0).sum(axis=1)
    df["PAY_RATIO"] = (paid / (owed + 1)).clip(0, 5)
    df["PAY_AMT_MEAN"] = df[PAYAMT].mean(axis=1)
    df["MAX_DELAY"] = df[PAY].max(axis=1)
    df["N_LATE_MONTHS"] = (df[PAY] > 0).sum(axis=1)
    df["N_REVOLVING_MONTHS"] = (df[PAY] == 0).sum(axis=1)
    df["DELAY_TREND"] = df["PAY_1"] - df[["PAY_4", "PAY_5", "PAY_6"]].mean(axis=1)
    return df


def prepare(df):
    df, clean_log = clean(df)
    return engineer(df), clean_log


def model_columns(df):
    drop = set(C.EXCLUDE_FROM_MODEL + [C.TARGET, "AGE_BAND"])
    return [c for c in df.columns if c not in drop]
