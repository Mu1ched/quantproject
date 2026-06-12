# quantproject

A quantitative forex trading system for researching, backtesting, and live-trading Opening Range Breakout (ORB) strategies across major forex pairs.

---

## Findings

This project is published primarily as a methodology case study. Over 150 ORB
strategy variants were tested across 5 major pairs under honest execution
modeling — live-measured raw-ECN spreads, per-lot commission, slippage, and
session-aware fill simulation — with multiple-testing corrections applied
(deflated Sharpe ratio, bootstrap confidence intervals, Benjamini-Hochberg FDR).

**Result: none survived realistic costs.** Edges that look strong under naive
backtesting (zero-cost fills, raw Sharpe) disappear once spread, commission,
and multiple-testing bias are accounted for. The validation machinery that
kills false positives — not a profitable strategy — is the deliverable here.

---

## What it does

| Component | File | Purpose |
|---|---|---|
| Edge Engine | `edge_engine.py` | Data loading, backtesting loop, risk management, hypothesis testing |
| Strategies | `edge_hypotheses.py` | Strategy entry functions and parameter grids |
| Research GUI | `edge_gui.py` | Streamlit dashboard for running parameter sweeps |
| Pattern Miner | `edge_miner.py` | Automated edge discovery using LightGBM + SHAP |
| Live Trader | `MT5live.2.py` | Real-time execution via MetaTrader 5 |
| Live Dashboard | `live_gui.py` | Real-time monitoring dashboard |

---

## Requirements

- Python 3.10+
- Windows (MetaTrader5 library is Windows-only)
- A MetaTrader 5 terminal with an active account
- Dukascopy tick data access (fetched automatically via HTTP)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root (never commit this):

```
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Server
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHANNEL=@your_channel

# Edge Discovery Agent
ANTHROPIC_API_KEY=your_anthropic_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_numeric_chat_id
```

---

## Running a backtest

```bash
streamlit run edge_gui.py
```

1. The dashboard loads and downloads Dukascopy tick data for all pairs (cached after first run)
2. Select a sweep from the sidebar
3. Click **Run Sweep** — results are written to `edge_results.db`
4. Switch to the **Results** tab to view metrics, equity curves, and hypothesis test output

---

## Running the live trader

```bash
python MT5live.2.py
```

Ensure MetaTrader 5 is open and logged in before starting. The script will:
- Connect to MT5 and verify the account
- Monitor session open times and place ORB trades automatically
- Post trade reasoning and results to Telegram

---

## Running the live dashboard

```bash
streamlit run live_gui.py
```

---

## Adding a new strategy

1. Define an entry function in `edge_hypotheses.py` with this signature:
   ```python
   def entry_my_strategy(bst, slot, row, ts, pair, slip, hspd,
                         sess_cfg, regime, regime_mult, **kwargs):
   ```
2. Use the building blocks from `edge_engine`: `spread_gate()`, `rv_size()`, `check_and_fill()`
3. Register a sweep at the bottom of `edge_hypotheses.py`:
   ```python
   SWEEPS['my_strategy'] = {
       'entry_fn': entry_my_strategy,
       'grid': ParameterGrid({...}),
       ...
   }
   ```
4. The sweep will appear in the research GUI automatically

---

## Autonomous Edge Discovery Agent

An agentic system that runs continuously, generates novel strategies via Claude, backtests them automatically, and sends you a daily top-5 report via Telegram.

### Setup

Add to `.env`:
```
ANTHROPIC_API_KEY=your_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Running

```bash
# Start the continuous loop (runs forever, Ctrl-C to stop)
python -m agent.main

# Test Claude generation without running a backtest
python -m agent.main --dry-run

# Send today's report immediately
python -m agent.main --report-now

# Run one round and exit (useful for testing)
python -m agent.main --round-once
```

### How it works

1. Every ~30 seconds: determines current session from UTC time (Asian / London / NY)
2. Calls Claude API to generate 3 novel entry functions targeting that session
3. Writes each to `agent/generated/entry_NAME.py` (importable, picklable)
4. Runs a full parameter sweep via `run_sweep()` — results written to `edge_results.db`
5. Scores survivors with composite metric (Sharpe, DSR, win rate, regime stability)
6. At 18:00 UTC: Claude writes a narrative, top-5 sent to Telegram
7. All generated strategies and results tracked in `agent/agent_results.db`

### Output

- `agent/agent_results.db` — scored survivor strategies
- `agent/generated/` — all generated Python strategy files
- `agent/agent.log` — rotating log (10MB × 3 files)
- Telegram message at 18:00 UTC daily

---

## Fake-edge protection

The engine applies four layers before declaring a result significant:

1. **Walk-forward CV** inside the miner
2. **Adversarial KS-test** — checks for distribution shift between train/test
3. **Deflated Sharpe + Bootstrap CI** — corrects for multiple testing bias
4. **Benjamini-Hochberg FDR** — controls false discovery rate across sweeps

---

## Data caching

- Downloaded tick data is cached to `edge_prepared_cache/` as Parquet files
- Delete this folder or pass `force_refresh=True` to `load_all_data()` to re-download
- Sweep results persist in `edge_results.db` (SQLite)

---

## Common issues

| Problem | Fix |
|---|---|
| MT5 connection fails | Ensure MT5 terminal is open and `AutoTrading` is enabled |
| Dukascopy fetch times out | Reduce `n_workers` in `load_all_data()` or retry — transient HTTP errors are normal |
| Parquet cache gives wrong columns | Delete `edge_prepared_cache/` and re-run |
| `ModuleNotFoundError: MetaTrader5` | This library is Windows-only; research GUI works without it |

---

## Project structure

```
quantproject/
├── edge_engine.py       # Core infrastructure (do not put strategies here)
├── edge_hypotheses.py   # Your strategy workspace
├── edge_gui.py          # Research dashboard
├── edge_miner.py        # Automated pattern discovery
├── MT5live.2.py         # Live trading engine
├── live_gui.py          # Live monitoring dashboard
├── requirements.txt
├── .env                 # Secrets — never commit
└── edge_prepared_cache/ # Auto-generated data cache
```
