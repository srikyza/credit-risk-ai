"""Single place for every setting. Change here, rerun `python run_pipeline.py`."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "UCI_Credit_Card.csv"      # .csv or .zip both work
MODEL_PATH = ROOT / "models" / "credit_risk_model.joblib"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
OUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"

RANDOM_STATE = 42
TARGET_RAW = "default.payment.next.month"
TARGET = "DEFAULT"

# 70 / 15 / 15 stratified split
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Responsible-AI choice: protected attributes are NOT used for prediction,
# but they ARE used afterwards to audit the model for unequal outcomes.
EXCLUDE_FROM_MODEL = ["ID", "SEX", "MARRIAGE"]
AUDIT_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE", "AGE_BAND"]

# Business cost assumption (edit to match your bank's economics):
# approving a client who defaults costs COST_FN, rejecting a good client costs COST_FP.
COST_FN = 5.0
COST_FP = 1.0

# Risk tiers on predicted probability of default (PD)
TIER_BINS = [0.0, 0.15, 0.35, 1.0]
TIER_LABELS = ["Low", "Medium", "High"]

# Drift monitoring: PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 retrain
PSI_WATCH = 0.10
PSI_ALERT = 0.25
