"""
ONE COMMAND, END TO END:

    python run_pipeline.py                      # ingest -> validate -> train -> evaluate -> monitor -> report
    python run_pipeline.py --data path/to.csv   # use another file (csv or zip)
    python run_pipeline.py --score new.csv      # score a fresh batch with the saved model (no retraining)

Exit code 0 = success, 1 = failure, so a scheduler (cron / Task Scheduler / GitHub Actions) can run it unattended.
"""
import argparse, json, logging, sys, time
from datetime import datetime
import joblib, pandas as pd
from creditrisk import config as C
from creditrisk import ingest, features, train, evaluate, monitor, score

log = logging.getLogger("pipeline")


def setup_logging():
    C.LOG_DIR.mkdir(exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(C.LOG_DIR / "pipeline.log", mode="w")])


def stage(name):
    log.info("=" * 8 + f" {name} " + "=" * 8)
    return time.time()


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def run_training(data_path):
    t0 = time.time()
    s = stage("1/6 INGEST + VALIDATE")
    raw = ingest.load_raw(data_path)
    dq = ingest.validate(raw)

    s = stage("2/6 CLEAN + FEATURE ENGINEERING")
    df, clean_log = features.prepare(raw)
    cols = features.model_columns(df)
    log.info("Cleaning actions: %s", clean_log)
    log.info("%d model features (excluded for fairness: %s)", len(cols), C.EXCLUDE_FROM_MODEL)

    s = stage("3/6 TRAIN + SELECT MODEL")
    tr, va, te = train.split(df)
    model, best, board = train.train_and_select(tr, va, cols)

    s = stage("4/6 EVALUATE + EXPLAIN + FAIRNESS AUDIT")
    m, p, thr = evaluate.evaluate(model, cols, va, te, best)
    fi = evaluate.plots(model, cols, te, p, thr, m)
    tiers = evaluate.tier_table(te, p)
    fair = evaluate.fairness_audit(te, p, thr)

    s = stage("5/6 DRIFT MONITORING")
    ref_p = model.predict_proba(tr[cols])[:, 1]
    num_cols = [c for c in cols if c != "EDUCATION"]
    normal = monitor.drift_report(tr, te, num_cols, ref_p, p)
    stress_f, _ = features.prepare(monitor.simulate_drift(raw.loc[te.index]))
    stress_p = model.predict_proba(stress_f[cols])[:, 1]
    drifted = monitor.drift_report(tr, stress_f, num_cols, ref_p, stress_p)
    monitor.plot_drift(normal, drifted)
    log.info("Normal batch alerts: %d | Stress batch alerts: %d",
             (normal.status == "ALERT").sum(), (drifted.status == "ALERT").sum())

    s = stage("6/6 SAVE ARTEFACTS + AUTO-REPORT")
    C.MODEL_PATH.parent.mkdir(exist_ok=True); C.OUT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "columns": cols, "threshold": thr, "name": best,
                 "trained_at": datetime.now().isoformat(timespec="seconds"), "metrics": m}, C.MODEL_PATH)
    scored = score.score_frame(raw.loc[te.index], score.load_artifact())
    scored["ACTUAL_DEFAULT"] = te[C.TARGET].values
    scored.to_csv(C.OUT_DIR / "scored_test_clients.csv", index=False)
    board.to_csv(C.REPORT_DIR / "model_leaderboard.csv", index=False)
    fair.to_csv(C.REPORT_DIR / "fairness_audit.csv", index=False)
    drifted.to_csv(C.REPORT_DIR / "drift_stress_batch.csv", index=False)
    summary = {"data_quality": dq, "cleaning": clean_log, "metrics": m,
               "tiers": tiers.reset_index().to_dict("records"),
               "top_features": fi.head(10).round(4).to_dict(),
               "drift_alerts_normal": int((normal.status == "ALERT").sum()),
               "drift_alerts_stress": int((drifted.status == "ALERT").sum()),
               "generated": datetime.now().isoformat(timespec="seconds")}
    (C.REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(summary, board, tiers, fair, drifted, fi)
    log.info("DONE in %.1fs  -> %s", time.time() - t0, C.REPORT_DIR / "summary.md")


def write_report(S, board, tiers, fair, drifted, fi):
    m, dq = S["metrics"], S["data_quality"]
    t = tiers.reset_index().rename(columns={"tier": "Tier"})
    top_fair = fair[fair.attribute.isin(["SEX", "AGE_BAND", "EDUCATION", "MARRIAGE"])].copy()
    txt = f"""# Credit Default Early-Warning System: Auto-Generated Report
_Generated {S['generated']} by `run_pipeline.py`. Nothing in this file is typed by hand._

## 1. Data quality
{dq['rows']:,} clients, {dq['columns']} columns, {dq['missing_values']} missing values, {dq['duplicate_ids']} duplicate IDs.
Default rate **{dq['default_rate']*100:.1f}%**. Cleaning actions: {S['cleaning']}.

## 2. Model selection (validation set)
{md_table(board, "{:.4f}")}

Selected: **{m['model']}**. Decision threshold **PD >= {m['threshold']}**, chosen on validation to minimise cost
(missed default costs {C.COST_FN:g}x a wrongly declined good client).

## 3. Final performance on untouched test set
| Metric | Value |
|---|---|
| ROC-AUC | {m['roc_auc']:.3f} |
| Gini | {m['gini']:.3f} |
| KS statistic | {m['ks']:.3f} |
| PR-AUC | {m['pr_auc']:.3f} |
| Brier score | {m['brier']:.3f} |
| Recall (defaults caught) | {m['recall']:.1%} |
| Precision | {m['precision']:.1%} |
| Cost vs. approving everyone | **{m['cost_saving_pct']:.1f}% lower** |

![ROC and PR](figures/roc_pr.png)
![Confusion](figures/confusion.png)

## 4. Risk tiers
{md_table(t, "{:.3f}")}

![Tiers](figures/risk_tiers.png)

## 5. What drives risk
![Importance](figures/feature_importance.png)

## 6. Fairness audit (attributes excluded from the model, audited afterwards)
{md_table(top_fair, "{:.3f}")}

## 7. Drift monitoring
Normal test batch: **{S['drift_alerts_normal']}** PSI alerts. Simulated economic-stress batch: **{S['drift_alerts_stress']}** alerts
(PSI watch >= {C.PSI_WATCH}, alert >= {C.PSI_ALERT}). An alert is the trigger to retrain.

![Drift](figures/drift.png)
"""
    (C.REPORT_DIR / "summary.md").write_text(txt)


def run_scoring(path):
    art = score.load_artifact()
    raw = pd.read_csv(path)
    out = score.score_frame(raw, art)
    C.OUT_DIR.mkdir(exist_ok=True)
    dest = C.OUT_DIR / "scored_new_batch.csv"; out.to_csv(dest, index=False)
    log.info("Scored %d clients with %s -> %s", len(out), art["name"], dest)
    log.info("Tier mix: %s", out.RISK_TIER.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="training data (.csv or .zip)")
    ap.add_argument("--score", default=None, help="CSV of new clients to score with the saved model")
    a = ap.parse_args()
    setup_logging()
    try:
        run_scoring(a.score) if a.score else run_training(a.data)
    except Exception:
        log.exception("PIPELINE FAILED"); sys.exit(1)
