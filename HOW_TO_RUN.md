# Quantproject — How to Run

One file. Two questions answered: **how do I start it**, and **where do I look at the results**.

---

## TL;DR

```
cd C:\Users\malac\Downloads\Quantproject
streamlit run edge_gui.py
```

Browser opens at `http://localhost:8501`. Everything else is buttons.

---

## First-time setup (do this once)

1. **Python 3.11+** installed.
2. **Dependencies** — from the project folder:
   ```
   pip install -r requirements.txt
   ```
   (If there's no `requirements.txt`, the key ones are: `streamlit pandas numpy matplotlib scipy scikit-learn anthropic python-dotenv requests lightgbm shap optuna`.)
3. **`.env` file** in the project root with at minimum:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_CHAT_ID=...
   ```
   Without these the autonomous agent refuses to start. The GUI still works without them — you just can't drive the agent loop.
4. **Data** — first time you click *Load / Refresh Data* in the sidebar, Dukascopy tick data is downloaded into `duka_cache/` (10–30 minutes; multi-GB).

---

## The GUI — what each tab does

Launch with `streamlit run edge_gui.py`. Sidebar: load data, then pick a sweep from the dropdown to populate the result browsers.

| Tab | What it does | When to use |
|---|---|---|
| **▶ Run** | Pick a sweep from `edge_hypotheses.py`, set workers + cost multiplier, run grid or Optuna search. | Manually testing one strategy family. |
| **📋 Results** | Sortable table of every hypothesis in the selected sweep. Filter by BH significance / DSR / CI lower bound. | Browsing what came out of a sweep. |
| **📈 Detail** | Equity curves (train + test), per-pair / per-exit-reason breakdown, hypothesis tests. | Deep-diving one strategy. |
| **🛡 Robustness** | Walk-forward folds, regime breakdown, Monte Carlo prop-firm pass rate, fake-edge filters. | Deciding whether a survivor is real. |
| **🧬 Mine** | Automated pattern miner (LightGBM + SHAP). Includes the self-improving loop that mines → sweeps → learns → mutates over N rounds. | Letting the system find conditions you didn't think of. |
| **📊 Portfolio** | Pairwise correlation between survivors in a sweep. Flags pairs > 0.70. | Before sizing a portfolio of survivors. |
| **💸 TCA** | Live Transaction Cost Analysis. Per-pair effective spread / slippage, by-hour cost profile, per-strategy live-vs-backtest decay. | After MT5 has been running long enough to log fills. |
| **🤖 Agent** | Start/stop the autonomous Claude loop. View recent strategies, log tail, generated reports. | Daily driver. Click *Start agent* and walk away. |

---

## Running the autonomous agent (the daily-driver path)

Two ways:

**A. From the GUI** — open the **🤖 Agent** tab.
- Pick mode: `continuous loop` (default) / `single round` / `dry run`.
- Click **▶ Start agent**.
- Status pill goes green; PID shows; log tail and recent-strategies table refresh on every Streamlit re-render.
- Click **⏹ Stop agent** to kill it.

**B. From the command line** (use this on the VPS where you don't want a browser running):
```
cd C:\Users\malac\Downloads\Quantproject
python -m agent.main                # continuous loop (default)
python -m agent.main --round-once   # one round, then exit
python -m agent.main --dry-run      # generate one hypothesis, no backtest
python -m agent.main --report-now   # send today's Telegram report immediately
python -m agent.main --analyse      # deep-dive recent survivors
```

Spend is gated by `BUDGET_DAILY_USD` and `BUDGET_TOTAL_USD` in `agent/config.py` — when hit, Claude calls return `None` instead of charging your card.

---

## Where to find results

| What | Where |
|---|---|
| **Database (everything)** | `agent/agent_results.db` — `tested_strategies`, `live_trades`, `live_executions`, `live_decay`, `budget_log` |
| **Sweep DB (legacy)** | `edge_results.db` (project root) — used by the *Run* tab |
| **Generated strategy code** | `agent/generated/` — one `.py` per strategy Claude writes |
| **Deep-dive markdown reports** | `agent/reports/<YYYY-MM-DD>/<strategy_name>/` — equity curves, robustness, regime breakdown |
| **TCA reports** | `agent/reports/<YYYY-MM-DD>/tca/tca_report.md` (refresh from the TCA tab or `python -m agent.tca`) |
| **Agent log** | `agent/agent.log` (rotates at 10 MB, keeps 3 backups) — the GUI Agent tab tails the last 80 lines |
| **MT5 live CSVs** (input) | Project root: `live_trade_log.csv`, `execution_quality.csv` — written by MT5live |

---

## The TCA workflow (live cost analysis)

Once MT5live has been running and writing CSVs:

1. **🤖 Agent** tab → confirm the agent has been ingesting (it auto-runs `live_ingest` every `LIVE_INGEST_EVERY_ROUNDS` rounds), or
2. **💸 TCA** tab → click **📥 Ingest latest MT5 CSVs** to force an immediate refresh.
3. Click **🔄 Refresh TCA + write report**.
4. Read the three sections:
   - **Per-pair summary**: `cost_per_trade_pips` — anything > 2 pips eats most retail strategies. Look at `fill_quality`; `tail-heavy` = occasional huge slippage (usually news).
   - **By-hour profile**: spike in `max_spread` at certain UTC hours = broker silently widening. Common at 21:00–22:00 (rollover).
   - **Per-strategy decay**: `verdict` column. `KILL` = stop running this strategy live. `PROMOTE` = live matches/exceeds backtest.
5. Open the markdown report from the expander at the bottom, or read directly at `agent/reports/<date>/tca/tca_report.md`.

Or skip the GUI:
```
python -m agent.tca
```

---

## Common operations

**Re-run on a fresh sweep without restarting the GUI** — pick a new sweep in the sidebar dropdown; all tabs re-bind.

**Check spend** — `agent/config.py` has the caps; the actual cumulative spend lives in `agent_results.db`'s `budget_log` table.

**Wipe and start over** — delete `agent/agent_results.db` and `edge_results.db`. The next agent round recreates them.

**Inspect Claude's prompts** — `agent/agent.log` logs every tool call with token counts.

**Run on a constrained VPS** — current `agent/config.py` is already tuned for an 8 GB Windows VPS: `BACKTEST_WORKERS = 1`, `HYPOTHESES_PER_BATCH = 2`, grid capped at 120 combos. Don't bump these without monitoring memory.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `MemoryError` mid-sweep | Check `BACKTEST_WORKERS = 1` in `agent/config.py`. Windows multiprocessing copies all data per worker. |
| GUI shows "Agent modules failed to import" | Run from the project root (`C:\Users\malac\Downloads\Quantproject`), not from inside `agent/`. |
| Agent starts but no reports appear | Filters are strict by design. Check `agent_results.db`: `SELECT verdict, COUNT(*) FROM tested_strategies GROUP BY verdict`. Most strategies are correctly rejected. |
| TCA tab empty | No live trades ingested. Either MT5 hasn't traded, or the CSVs aren't where `LIVE_LOG_DIR` (project root) expects. |
| Telegram report never arrives | `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` in `.env`. Test with `python -m agent.main --report-now`. |
| VPS becomes unresponsive | Hard reboot via Contabo control panel → Manage → Control. Then kill stray `python.exe` and restart. |
