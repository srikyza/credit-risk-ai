# Credit Default Early-Warning System: an end-to-end automated AI solution for Banking

**Vertical:** retail banking / credit risk. **Data:** UCI "Default of Credit Card Clients" (Taiwan, 30,000 clients, 6 months of history).
**Business question:** which cardholders will miss their payment next month, and what should the bank do about it?

## Run it (one command)
```bash
pip install -r requirements.txt
python run_pipeline.py                 # full pipeline, about 30 seconds, writes everything below
streamlit run app.py                   # dashboard
python run_pipeline.py --score new.csv # score a fresh batch with the saved model, no retraining
```
Google Colab: upload the folder, then `!pip install -r requirements.txt` and `!python run_pipeline.py`.

## Pipeline stages (each is a module in `creditrisk/`, orchestrated by `run_pipeline.py`)
| # | Stage | Module | What it automates |
|---|---|---|---|
| 1 | Ingest + validate | `ingest.py` | schema check, range checks, duplicate/missing report; stops on bad data |
| 2 | Clean + features | `features.py` | fixes undocumented codes, builds utilisation, payment-ratio, delay-trend features (stateless, so no leakage) |
| 3 | Train + select | `train.py` | stratified 70/15/15 split, 3 models, tuned gradient boosting, selection on validation AUC |
| 4 | Evaluate | `evaluate.py` | cost-based threshold, ROC/PR/KS/Gini, permutation importance, risk tiers, **fairness audit** |
| 5 | Monitor | `monitor.py` | PSI drift per feature and on the score; simulated economic-stress batch to prove alerts fire |
| 6 | Deploy / score | `score.py`, `app.py` | saved model artefact, batch scoring CLI, Streamlit dashboard |
| 7 | Report | `run_pipeline.py` | auto-writes `reports/summary.md`, `metrics.json`, figures, scored CSVs |


- **Imbalance (22% defaults):** handled with stratified splits, PR-AUC, and a *cost-based decision threshold* (a missed default costs 5x a wrongly declined client, editable in `config.py`) instead of resampling, which keeps probabilities honest. Result: predicted PD matches observed default rate in every tier.
- **Responsible AI:** SEX and MARRIAGE are excluded from the model; they are used only afterwards in the fairness audit.
- **No leakage:** the test set is touched once. Threshold and model choice use validation only.
- **Monitoring:** PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 retrain.


- Linux/Mac cron, retrain weekly: `0 2 * * 1 cd /path/credit_risk_ai && python run_pipeline.py`
- Windows: Task Scheduler running `python run_pipeline.py`. The exit code is 0 on success and 1 on failure.
- Extension: drop new monthly CSVs in `data/`, run `--score`, and trigger a retrain when `drift_stress_batch.csv` style alerts appear.

## Known limitations 
- Single snapshot from 2005 in Taiwan, with no time ordering, so true out-of-time validation is not possible.
- PD estimates reflect that population; re-calibrate before using on another market.
- Cost ratio 5:1 is an assumption, not bank data.
