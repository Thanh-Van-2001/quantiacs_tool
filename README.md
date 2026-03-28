# Quantiacs Alpha Factory

> Auto-generate, optimize & submit alpha trading strategies to Quantiacs competitions.

Two tools in one repo:
- **`q24_optimizer.py`** — Optuna-powered optimizer for the Q24 Crypto contest (recommended)
- **`quantiacs_alpha_factory.py`** — Grid-search generator for all contest types (futures, stocks, crypto)

---

## Quick Start

```bash
# 1. Setup environment
conda create -n qntdev -c conda-forge "python>=3.11,<3.14" ipykernel ta-lib "quantiacs-source::qnt" dash
conda activate qntdev
pip install optuna

# 2. Set API key (get from https://quantiacs.com/personalpage/homepage)
# PowerShell
$env:API_KEY = "your_api_key"
# Linux/Mac
export API_KEY="your_api_key"

# 3. Run optimizer
python q24_optimizer.py --n-trials 100
```

---

## Tool 1: Q24 Crypto Optimizer (`q24_optimizer.py`)

Targeted at the **Q24 Crypto Top-10 Long-Only Contest**. Uses Optuna TPE sampler + single-pass backtesting for speed.

### How it works

1. Loads crypto daily data **once** into memory
2. Optuna samples an alpha template + parameters each trial
3. Single-pass backtest runs in **2-5 seconds** (vs 3-5 min multi-pass)
4. Results stored in SQLite — **resume-safe** (Ctrl+C and rerun anytime)
5. Top strategies auto-exported as submission-ready `strategy.ipynb`

### Commands

```bash
# Run 100 optimization trials, export top 5
python q24_optimizer.py --n-trials 100 --top-k 5

# Run 300 more trials (resumes from previous)
python q24_optimizer.py --n-trials 300

# View results without running new trials
python q24_optimizer.py --mode results

# Export a specific trial as submission
python q24_optimizer.py --mode export --trial 42
```

### Alpha Templates (8 strategies)

| Template | Description |
|----------|-------------|
| `sma_crossover` | Dual SMA crossover + RSI filter (from Q24 official template) |
| `ema_momentum` | EMA trend x momentum sizing |
| `triple_sma` | Triple SMA alignment (fast > mid > slow) |
| `rsi_momentum` | RSI sweet-spot + SMA trend filter |
| `breakout_long` | Donchian channel breakout + EMA filter |
| `vol_weighted` | Inverse volatility weighting + trend |
| `macd_long` | MACD histogram positive = long |
| `combined_score` | Multi-factor: trend + momentum + RSI |

Optuna automatically selects the best template AND tunes its parameters.

### Q24 Contest Rules

- Long-only positions on top-10 crypto by market cap
- In-sample period starts 2016-01-01
- Minimum Sharpe Ratio: **1.0**
- No manual asset selection (must be automatic via `is_liquid`)
- Top 7 unique-user strategies by Sharpe win
- Only Quantiacs-provided data allowed

### Output Structure

```
submissions/
  sma_crossover_a1b2c3d4/
    strategy.ipynb    <- Upload this to quantiacs.com
    strategy.py       <- Local reference
    init.ipynb        <- Dependencies (empty)
    metadata.json     <- Alpha name, params, metrics
  combined_score_e5f6g7h8/
    ...
q24_crypto.db         <- Optuna study (all trial history)
```

---

## Tool 2: Alpha Factory (`quantiacs_alpha_factory.py`)

Grid-search alpha generator supporting all Quantiacs contests.

### Commands

```bash
# List templates for a contest
python quantiacs_alpha_factory.py --mode list --competition futures

# Generate strategies (no backtest)
python quantiacs_alpha_factory.py --mode generate --competition stocks --num-alphas 30

# Full pipeline: generate -> backtest -> rank -> export
python quantiacs_alpha_factory.py --mode pipeline --competition cryptodaily --num-alphas 50 --top-k 5

# Backtest a single strategy
python quantiacs_alpha_factory.py --mode backtest --strategy path/to/strategy.py

# Prepare submission package
python quantiacs_alpha_factory.py --mode submit --strategy path/to/strategy.py
```

### Supported Contests

| Competition | Flag | Assets |
|-------------|------|--------|
| Futures | `--competition futures` | 70+ global futures |
| Stocks | `--competition stocks` | NASDAQ-100 |
| Crypto Daily | `--competition cryptodaily` | Top-10 crypto |
| Crypto Futures | `--competition cryptofutures` | Bitcoin futures |

### Alpha Templates (16 strategies, 219+ parameter combos)

**Trend**: `sma_crossover`, `ema_crossover`, `triple_ema`, `trix_signal`
**Mean Reversion**: `bollinger_reversion`, `rsi_reversion`, `zscore_reversion`
**Momentum**: `momentum_rank`, `rate_of_change`
**Volatility**: `atr_breakout`, `vol_adjusted_momentum`
**Combo**: `macd_rsi_combo`, `ema_rsi_atr_combo`
**Breakout**: `channel_breakout`, `range_compression`
**Long-Only**: `momentum_long_only`, `equal_weight_liquid`

---

## Adding Custom Alphas

### For Q24 Optimizer

Add to the `ALPHAS` dict in `q24_optimizer.py`:

```python
def alpha_my_custom(data, params):
    close = data.sel(field="close")
    is_liquid = data.sel(field="is_liquid")
    # your logic here
    return weights * is_liquid

ALPHAS["my_custom"] = {
    "func": alpha_my_custom,
    "suggest": lambda t: {
        "param1": t.suggest_int("param1", 5, 50),
        "param2": t.suggest_float("param2", 0.1, 0.9),
    },
}
```

Then add a matching `elif alpha_name == "my_custom"` block in `generate_strategy_code()`.

### For Alpha Factory

Use the `@register_alpha` decorator — see `examples/custom_alpha.py`.

---

## Submission Workflow

1. Run optimizer -> strategies exported to `submissions/`
2. Go to https://quantiacs.com/personalpage/strategies
3. Upload `strategy.ipynb` to Quantiacs Jupyter
4. Run all cells to verify
5. Click **Submit**

### Tips

- **Avoid overfitting**: Compare single-pass vs multi-pass Sharpe. If they match, no forward-looking bias.
- **Diversify**: Submit strategies from different alpha templates.
- **Simple wins**: Simple strategies generalize better out-of-sample.
- **Resume**: Optuna study persists in SQLite. Ctrl+C and rerun to continue.
- **Template copies won't win** — always customize parameters.

---

## Comparison

| | Alpha Factory (v1) | Q24 Optimizer (v2) |
|---|---|---|
| Speed | ~3-5 min/strategy (multi-pass subprocess) | **~2-5 sec/strategy (single-pass in-memory)** |
| Optimization | Grid search (brute force) | **Optuna TPE** (Bayesian) |
| Data loading | Reload per strategy | **Load once, cache** |
| Resume | No | **SQLite, resume-safe** |
| Scope | All contests | Q24 Crypto Top-10 |
| Sharpe target | >= 0.7 | >= 1.0 |

---

## Project Structure

```
quantiacs_alpha_factory/
  q24_optimizer.py              # Optuna optimizer for Q24 Crypto (recommended)
  quantiacs_alpha_factory.py    # Grid-search generator for all contests
  README.md
  examples/
    quick_start.py
    custom_alpha.py
  submissions/                  # Auto-generated submission packages
    {alpha}_{hash}/
      strategy.ipynb
      strategy.py
      init.ipynb
      metadata.json
  q24_crypto.db                 # Optuna study database
```

## License

MIT
