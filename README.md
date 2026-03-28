# Quantiacs Alpha Factory v1.0

> Auto-generate, backtest & submit alpha trading strategies to Quantiacs competitions.

Built for quant researchers who want to systematically explore alpha space on the Quantiacs platform.

## Architecture

```
quantiacs_alpha_factory/
├── quantiacs_alpha_factory.py   # Main tool (all-in-one)
├── README.md                     # This file
├── examples/
│   ├── quick_start.py            # Quick start example
│   └── custom_alpha.py           # How to add custom alphas
└── quantiacs_workspace/          # Generated at runtime
    ├── generated_alphas/         # All generated .py strategy files
    ├── submissions/              # Submission-ready packages
    └── alpha_report.json         # Results ranking
```

## Prerequisites

```bash
# Install Quantiacs toolbox
pip install git+https://github.com/quantiacs/toolbox.git

# Set your API key
export QUANTIACS_API_KEY="your_api_key_here"
```

## Quick Start

### 1. List available alpha templates
```bash
python quantiacs_alpha_factory.py --mode list --competition futures
```

### 2. Generate alphas (no backtest, fast)
```bash
python quantiacs_alpha_factory.py --mode generate --competition futures --num-alphas 30
```

### 3. Full pipeline: generate → backtest → rank → submit
```bash
python quantiacs_alpha_factory.py --mode pipeline \
    --competition futures \
    --num-alphas 50 \
    --top-k 5 \
    --api-key YOUR_KEY
```

### 4. Backtest a single strategy
```bash
python quantiacs_alpha_factory.py --mode backtest --strategy path/to/strategy.py
```

### 5. Prepare submission package
```bash
python quantiacs_alpha_factory.py --mode submit \
    --strategy path/to/strategy.py \
    --api-key YOUR_KEY
```

## Competition Types

| Type          | Flag            | Assets               |
|---------------|-----------------|----------------------|
| Futures       | `--competition futures`      | 70+ global futures   |
| Stocks        | `--competition stocks`       | NASDAQ-100 stocks    |
| Crypto Daily  | `--competition cryptodaily`  | Top-10 crypto        |
| Crypto Futures| `--competition cryptofutures`| Bitcoin futures      |

## Alpha Categories

The tool includes **14 alpha templates** across 7 categories:

### Trend Following
- `sma_crossover` — Dual SMA crossover (5 × 4 = 20 combos)
- `ema_crossover` — Dual EMA crossover
- `triple_ema` — Triple EMA alignment system
- `trix_signal` — TRIX indicator trend-following

### Mean Reversion
- `bollinger_reversion` — Bollinger Band reversion
- `rsi_reversion` — RSI oversold/overbought
- `zscore_reversion` — Z-Score rolling reversion

### Momentum
- `momentum_rank` — Cross-sectional momentum ranking
- `rate_of_change` — Rate of Change momentum

### Volatility
- `atr_breakout` — ATR-based breakout
- `vol_adjusted_momentum` — Volatility-adjusted (risk parity)

### Combo (Multi-Indicator)
- `macd_rsi_combo` — MACD + RSI combined signal
- `ema_rsi_atr_combo` — EMA trend + RSI filter + ATR sizing

### Breakout
- `channel_breakout` — Donchian Channel breakout
- `range_compression` — Range compression → expansion

### Long-Only (Crypto contests)
- `momentum_long_only` — Momentum-weighted allocation
- `equal_weight_liquid` — Equal-weight with optional trend filter

## Adding Custom Alphas

```python
from quantiacs_alpha_factory import register_alpha

@register_alpha(
    name="my_custom_alpha",
    category="custom",
    description="My proprietary signal",
    competition_types=["futures", "stocks"],
    param_grid={
        "param1": [10, 20, 30],
        "param2": [0.5, 1.0],
    }
)
def gen_my_custom_alpha(params, competition_type):
    p1 = params["param1"]
    p2 = params["param2"]
    return f'''
import qnt.ta as qnta
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.futures_load_data(tail=period)

def strategy(data):
    close = data.sel(field="close")
    # Your custom logic here
    weights = ...
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={p1 + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''
```

## Quantiacs Competition Rules (Key Points)

1. **Sharpe ≥ 0.7** — In-sample Sharpe ratio must be at least 0.7
2. **No forward-looking bias** — Don't use future data in past decisions
3. **File must be `strategy.ipynb`** — Or import from strategy.py
4. **Weights for all trading days** — No gaps in output
5. **Timeout**: Futures = 10 min, Crypto = 5 min
6. **Max 50 strategies**, select 15 for contest
7. **Template copies are NOT eligible** — Always customize

## Submission Workflow

1. Tool generates a submission package with:
   - `strategy.ipynb` (ready for Quantiacs Jupyter)
   - `strategy.py` (local reference)
   - `init.ipynb` (external dependencies)
   - `precheck.ipynb` (pre-submission validation)

2. Upload to Quantiacs:
   - Go to https://quantiacs.com/personalpage/strategies
   - Upload the files to Jupyter environment
   - Run all cells to verify
   - Click "Submit" button

## Tips for Winning

- **Avoid overfitting**: Use multi-pass backtesting
- **Diversify**: Submit strategies across different categories
- **Low correlation**: Strategies should be uncorrelated with each other
- **Simple > Complex**: Simple strategies often generalize better
- **Use `precheck.ipynb`** before every submission
- **Monitor live performance** after submission

## License

MIT — Use freely for your quant research.
