"""Credit-risk dashboard.  Run:  streamlit run app.py   (after `python run_pipeline.py`)"""
import json
import pandas as pd, streamlit as st
from creditrisk import config as C
from creditrisk.score import load_artifact, score_frame

st.set_page_config(page_title="Credit Risk AI", page_icon="🏦", layout="wide")
st.title("🏦 Credit Default Early-Warning System")

if not C.MODEL_PATH.exists():
    st.error("No trained model found. Run `python run_pipeline.py` first."); st.stop()

art = load_artifact()
S = json.loads((C.REPORT_DIR / "metrics.json").read_text())
m = S["metrics"]
st.caption(f"Model: **{art['name']}**  |  trained {art['trained_at']}  |  decision threshold PD ≥ {art['threshold']}")

t1, t2, t3, t4 = st.tabs(["📊 Portfolio & model", "🧍 Score one client", "📁 Score a batch", "🛡️ Monitoring & fairness"])

with t1:
    c = st.columns(5)
    c[0].metric("ROC-AUC", f"{m['roc_auc']:.3f}"); c[1].metric("KS", f"{m['ks']:.3f}")
    c[2].metric("Defaults caught", f"{m['recall']:.0%}"); c[3].metric("Precision", f"{m['precision']:.0%}")
    c[4].metric("Cost vs no model", f"-{m['cost_saving_pct']:.0f}%")
    a, b = st.columns(2)
    a.image(str(C.FIG_DIR / "roc_pr.png")); b.image(str(C.FIG_DIR / "risk_tiers.png"))
    a.image(str(C.FIG_DIR / "feature_importance.png")); b.image(str(C.FIG_DIR / "confusion.png"))
    st.subheader("Model leaderboard"); st.dataframe(pd.read_csv(C.REPORT_DIR / "model_leaderboard.csv"), hide_index=True)

with t2:
    st.write("Enter a client's profile. Bill and payment amounts are the typical monthly figure over the last 6 months.")
    c1, c2, c3 = st.columns(3)
    limit = c1.number_input("Credit limit (NT$)", 10000, 1000000, 150000, 10000)
    age = c2.number_input("Age", 18, 90, 35)
    edu = c3.selectbox("Education", [1, 2, 3, 4], format_func=lambda x: {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}[x])
    bill = c1.number_input("Typical monthly bill (NT$)", 0, 1000000, 60000, 5000)
    paid = c2.number_input("Typical monthly payment (NT$)", 0, 1000000, 5000, 1000)
    st.write("Repayment status per month (-2 no use, -1 paid in full, 0 revolving, 1+ = months late). Most recent first:")
    cols = st.columns(6)
    pays = [cols[i].selectbox(f"Month -{i+1}", list(range(-2, 9)), index=2, key=f"p{i}") for i in range(6)]
    if st.button("Score this client", type="primary"):
        row = {"ID": 1, "LIMIT_BAL": limit, "SEX": 1, "EDUCATION": edu, "MARRIAGE": 1, "AGE": age}
        row.update({f"PAY_{i+1}": pays[i] for i in range(6)})
        row.update({f"BILL_AMT{i}": bill for i in range(1, 7)}); row.update({f"PAY_AMT{i}": paid for i in range(1, 7)})
        r = score_frame(pd.DataFrame([row]), art).iloc[0]
        st.metric("Probability of default next month", f"{r.PROB_DEFAULT:.1%}")
        (st.success if r.RISK_TIER == "Low" else st.warning if r.RISK_TIER == "Medium" else st.error)(
            f"Risk tier: {r.RISK_TIER}  ->  {r.DECISION}")

with t3:
    up = st.file_uploader("Upload CSV in the original UCI format", type="csv")
    if up:
        out = score_frame(pd.read_csv(up), art)
        st.dataframe(out.head(200), hide_index=True)
        st.bar_chart(out.RISK_TIER.value_counts().reindex(C.TIER_LABELS))
        st.download_button("Download scored file", out.to_csv(index=False), "scored_batch.csv", "text/csv")

with t4:
    st.subheader("Drift monitor")
    st.write(f"PSI alerts on a normal batch: **{S['drift_alerts_normal']}**. On a simulated economic-stress batch: **{S['drift_alerts_stress']}** -> retrain trigger.")
    st.image(str(C.FIG_DIR / "drift.png"))
    st.subheader("Fairness audit (protected attributes excluded from model, audited afterwards)")
    st.dataframe(pd.read_csv(C.REPORT_DIR / "fairness_audit.csv").round(3), hide_index=True)
