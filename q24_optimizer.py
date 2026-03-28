#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║         QUANTIACS Q24 CRYPTO ALPHA OPTIMIZER v2.0                  ║
║  Optuna-powered alpha search + single-pass backtest + auto-submit  ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
    python q24_optimizer.py                        # Run full optimization (100 trials)
    python q24_optimizer.py --n-trials 200         # More trials
    python q24_optimizer.py --mode export --trial 0 # Export best trial as submission
"""

import os
import sys
import json
import warnings
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

warnings.filterwarnings("ignore")

# ── Quantiacs imports ──────────────────────────────────────────────
import xarray as xr
import numpy as np
import pandas as pd

import qnt.stats as qnstats
import qnt.data as qndata
import qnt.output as qnout
import qnt.ta as qnta

# ── Optuna ─────────────────────────────────────────────────────────
try:
    import optuna
    from optuna.trial import Trial
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("❌ Optuna not installed. Run: pip install optuna")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
#  GLOBAL DATA CACHE (load once, reuse across all trials)
# ═══════════════════════════════════════════════════════════════════

_DATA_CACHE = {}

def get_data(min_date: str = "2015-01-01") -> xr.DataArray:
    """Load crypto daily data once, cache globally."""
    if "data" not in _DATA_CACHE:
        print("📥 Loading crypto daily data (one-time)...")
        _DATA_CACHE["data"] = qndata.cryptodaily_load_data(min_date=min_date)
        print(f"   ✅ Loaded: {_DATA_CACHE['data'].shape}")
    return _DATA_CACHE["data"]


def get_benchmark(min_date: str = "2016-01-01") -> xr.DataArray:
    """Load benchmark weights once, cache globally."""
    if "benchmark" not in _DATA_CACHE:
        print("📥 Loading CRYPTO10 benchmark...")
        _DATA_CACHE["benchmark"] = qndata.index_load_weights(
            index_name="CRYPTO10", min_date=min_date
        )
        print(f"   ✅ Loaded benchmark")
    return _DATA_CACHE["benchmark"]


# ═══════════════════════════════════════════════════════════════════
#  ALPHA STRATEGY LIBRARY
#  Each function takes (data, params) → weights
# ═══════════════════════════════════════════════════════════════════

def alpha_sma_crossover(data: xr.DataArray, params: Dict) -> xr.DataArray:
    """SMA crossover with RSI filter (from Q24 template)."""
    close = data.sel(field="close")
    is_liquid = data.sel(field="is_liquid")

    sma_fast = qnta.sma(close, params["sma_fast"])
    sma_slow = qnta.sma(close, params["sma_slow"])
    rsi = qnta.rsi(close, params["rsi_period"])

    signal = xr.where(sma_fast > sma_slow, 1, 0) * is_liquid
    rsi_filter = xr.where(
        (rsi < params["rsi_lower"]) | (rsi > params["rsi_upper"]), 1, 0
    )
    return signal * rsi_filter


def alpha_ema_momentum(data: xr.DataArray, params: Dict) -> xr.DataArray:
    """EMA trend + momentum score for allocation sizing."""
    close = data.sel(field="close")
    is_liquid = data.sel(field="is_liquid")

    ema = qnta.ema(close, params["ema_period"])
    trend = xr.where(close > ema, 1, 0)

    # Momentum-based sizing
    mom = close / close.shift(time=params["mom_lookback"]) - 1
    mom_positive = mom.where(mom > 0).fillna(0)

    weights = trend * mom_positive * is_liquid
    return weights


def alpha_triple_sma(data: xr.DataArray, params: Dict) -> xr.DataArray:
    """Triple SMA alignment — only long when fast > mid > slow."""
    close = data.sel(field="close")
    is_liquid = data.sel(field="is_liquid")

    sma_f = qnta.sma(close, params["fast"])
    sma_m = qnta.sma(close, params["mid"])
    sma_s = qnta.sma(close, params["slow"])

    aligned = (sma_f > sma_m) & (sma_m > sma_s)
    weights = xr.where(aligned, 1, 0) * is_liquid
    return weights


def alpha_rsi_momentum(data: xr.DataArray, params: Dict) -> xr.DataArray:
    """RSI as a momentum signal — long when RSI in sweet spot."""
    close = data.sel(field="close")
    is_liquid = data.sel(field="is_liquid")

    rsi = qnta.rsi(close, params["rsi_period"])
    sma = qnta.sma(close, params["trend_sma"])

    trend_ok = close > sma
    rsi_ok = (rsi > params["rsi_entry"]) & (rsi < params["rsi_exit"])

    weights = xr.where(trend_ok & rsi_ok, 1, 0) * is_liquid
    return weights


def alpha_breakout_long(data: xr.DataArray, params: Dict) -> xr.DataArray:
    """Donchian channel breakout — long on new highs."""
    close = data.sel(field="close")
    high = data.sel(field="high")
    is_liquid = data.sel(field="is_liquid")

    upper = high.rolling(time=params["channel"]).max()
    ema_filter = qnta.ema(close, params["ema_filter"])

    breakout = (close >= upper) & (close > ema_filter)
    weights = xr.where(breakout, 1, 0) * is_liquid
    return weights


def alpha_vol_weighted(data: xr.DataArray, params: Dict) -> xr.DataArray:
    """Inverse volatility weighting with trend filter."""
    close = data.sel(field="close")
    is_liquid = data.sel(field="is_liquid")

    returns = close / close.shift(time=1) - 1
    vol = returns.rolling(time=params["vol_window"]).std()
    inv_vol = (1.0 / vol.where(vol > 1e-10)).fillna(0)

    ema = qnta.ema(close, params["trend_ema"])
    trend = xr.where(close > ema, 1, 0)

    weights = inv_vol * trend * is_liquid
    return weights


def alpha_macd_long(data: xr.DataArray, params: Dict) -> xr.DataArray:
    """MACD histogram positive → long."""
    close = data.sel(field="close")
    is_liquid = data.sel(field="is_liquid")

    macd_line, signal_line, histogram = qnta.macd(
        close, params["fast"], params["slow"], params["signal"]
    )

    weights = xr.where(histogram > 0, 1, 0) * is_liquid
    return weights


def alpha_combined_score(data: xr.DataArray, params: Dict) -> xr.DataArray:
    """Multi-factor scoring: trend + momentum + mean-reversion filter."""
    close = data.sel(field="close")
    is_liquid = data.sel(field="is_liquid")

    # Factor 1: EMA trend
    ema = qnta.ema(close, params["ema_period"])
    trend_score = xr.where(close > ema, 1, 0)

    # Factor 2: Momentum
    mom = close / close.shift(time=params["mom_period"]) - 1
    mom_score = xr.where(mom > 0, 1, 0)

    # Factor 3: RSI not overbought (avoid buying tops)
    rsi = qnta.rsi(close, params["rsi_period"])
    rsi_score = xr.where(rsi < params["rsi_cap"], 1, 0)

    # Combined
    total = trend_score + mom_score + rsi_score
    weights = xr.where(total >= params["min_score"], 1, 0) * is_liquid
    return weights


# ═══════════════════════════════════════════════════════════════════
#  ALPHA REGISTRY
# ═══════════════════════════════════════════════════════════════════

ALPHAS = {
    "sma_crossover": {
        "func": alpha_sma_crossover,
        "suggest": lambda t: {
            "sma_fast": t.suggest_int("sma_fast", 5, 30),
            "sma_slow": t.suggest_int("sma_slow", 20, 80),
            "rsi_period": t.suggest_int("rsi_period", 7, 21),
            "rsi_lower": t.suggest_int("rsi_lower", 20, 40),
            "rsi_upper": t.suggest_int("rsi_upper", 60, 85),
        },
    },
    "ema_momentum": {
        "func": alpha_ema_momentum,
        "suggest": lambda t: {
            "ema_period": t.suggest_int("ema_period", 10, 60),
            "mom_lookback": t.suggest_int("mom_lookback", 5, 30),
        },
    },
    "triple_sma": {
        "func": alpha_triple_sma,
        "suggest": lambda t: {
            "fast": t.suggest_int("fast", 5, 15),
            "mid": t.suggest_int("mid", 15, 40),
            "slow": t.suggest_int("slow", 40, 100),
        },
    },
    "rsi_momentum": {
        "func": alpha_rsi_momentum,
        "suggest": lambda t: {
            "rsi_period": t.suggest_int("rsi_period", 7, 21),
            "trend_sma": t.suggest_int("trend_sma", 20, 60),
            "rsi_entry": t.suggest_int("rsi_entry", 40, 60),
            "rsi_exit": t.suggest_int("rsi_exit", 70, 90),
        },
    },
    "breakout_long": {
        "func": alpha_breakout_long,
        "suggest": lambda t: {
            "channel": t.suggest_int("channel", 10, 55),
            "ema_filter": t.suggest_int("ema_filter", 20, 80),
        },
    },
    "vol_weighted": {
        "func": alpha_vol_weighted,
        "suggest": lambda t: {
            "vol_window": t.suggest_int("vol_window", 10, 40),
            "trend_ema": t.suggest_int("trend_ema", 20, 80),
        },
    },
    "macd_long": {
        "func": alpha_macd_long,
        "suggest": lambda t: {
            "fast": t.suggest_int("fast", 8, 16),
            "slow": t.suggest_int("slow", 20, 35),
            "signal": t.suggest_int("signal", 5, 12),
        },
    },
    "combined_score": {
        "func": alpha_combined_score,
        "suggest": lambda t: {
            "ema_period": t.suggest_int("ema_period", 15, 60),
            "mom_period": t.suggest_int("mom_period", 5, 30),
            "rsi_period": t.suggest_int("rsi_period", 7, 21),
            "rsi_cap": t.suggest_int("rsi_cap", 65, 85),
            "min_score": t.suggest_int("min_score", 2, 3),
        },
    },
}


# ═══════════════════════════════════════════════════════════════════
#  SINGLE-PASS BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════

def fast_backtest(
    data: xr.DataArray,
    alpha_name: str,
    params: Dict,
    is_start: str = "2016-01-01",
) -> Dict[str, float]:
    """Run single-pass backtest, return metrics dict."""
    try:
        alpha_func = ALPHAS[alpha_name]["func"]
        weights = alpha_func(data, params)
        weights = qnout.clean(weights, data, "crypto_daily_long")

        stats = qnstats.calc_stat(
            data, weights.sel(time=slice(is_start, None))
        ).sel(time=slice(is_start, None))

        last = stats.isel(time=-1)
        sharpe = float(last.sel(field="sharpe_ratio").values)
        ret = float(last.sel(field="mean_return").values)
        vol = float(last.sel(field="volatility").values)
        dd = float(last.sel(field="max_drawdown").values)

        return {
            "sharpe": sharpe if not np.isnan(sharpe) else -999,
            "return": ret if not np.isnan(ret) else 0,
            "volatility": vol if not np.isnan(vol) else 0,
            "max_drawdown": dd if not np.isnan(dd) else 0,
            "status": "ok",
        }
    except Exception as e:
        return {"sharpe": -999, "return": 0, "volatility": 0,
                "max_drawdown": 0, "status": f"error: {e}"}


# ═══════════════════════════════════════════════════════════════════
#  OPTUNA OBJECTIVE
# ═══════════════════════════════════════════════════════════════════

def create_objective(data: xr.DataArray):
    """Create Optuna objective that picks alpha + params and maximizes Sharpe."""

    def objective(trial: Trial) -> float:
        # Pick which alpha to use
        alpha_name = trial.suggest_categorical("alpha", list(ALPHAS.keys()))
        alpha_info = ALPHAS[alpha_name]

        # Suggest params for that alpha
        params = alpha_info["suggest"](trial)

        # Validate param constraints
        if alpha_name == "sma_crossover" and params.get("sma_fast", 0) >= params.get("sma_slow", 1):
            return -999
        if alpha_name == "triple_sma":
            if not (params.get("fast", 0) < params.get("mid", 0) < params.get("slow", 0)):
                return -999
        if alpha_name == "macd_long" and params.get("fast", 0) >= params.get("slow", 1):
            return -999
        if alpha_name == "rsi_momentum":
            if params.get("rsi_entry", 0) >= params.get("rsi_exit", 1):
                return -999

        # Run backtest
        result = fast_backtest(data, alpha_name, params)

        # Store extra info
        trial.set_user_attr("alpha_name", alpha_name)
        trial.set_user_attr("params", params)
        trial.set_user_attr("metrics", result)

        return result["sharpe"]

    return objective


# ═══════════════════════════════════════════════════════════════════
#  SUBMISSION GENERATOR
# ═══════════════════════════════════════════════════════════════════

def generate_strategy_code(alpha_name: str, params: Dict) -> str:
    """Generate standalone strategy.py code for Quantiacs submission."""

    # Build the strategy function body based on alpha_name
    if alpha_name == "sma_crossover":
        strategy_body = f"""
    sma_fast = qnta.sma(close, {params['sma_fast']})
    sma_slow = qnta.sma(close, {params['sma_slow']})
    rsi = qnta.rsi(close, {params['rsi_period']})
    signal = xr.where(sma_fast > sma_slow, 1, 0) * is_liquid
    rsi_filter = xr.where((rsi < {params['rsi_lower']}) | (rsi > {params['rsi_upper']}), 1, 0)
    return signal * rsi_filter"""

    elif alpha_name == "ema_momentum":
        strategy_body = f"""
    ema = qnta.ema(close, {params['ema_period']})
    trend = xr.where(close > ema, 1, 0)
    mom = close / close.shift(time={params['mom_lookback']}) - 1
    mom_positive = mom.where(mom > 0).fillna(0)
    return trend * mom_positive * is_liquid"""

    elif alpha_name == "triple_sma":
        strategy_body = f"""
    sma_f = qnta.sma(close, {params['fast']})
    sma_m = qnta.sma(close, {params['mid']})
    sma_s = qnta.sma(close, {params['slow']})
    aligned = (sma_f > sma_m) & (sma_m > sma_s)
    return xr.where(aligned, 1, 0) * is_liquid"""

    elif alpha_name == "rsi_momentum":
        strategy_body = f"""
    rsi = qnta.rsi(close, {params['rsi_period']})
    sma = qnta.sma(close, {params['trend_sma']})
    trend_ok = close > sma
    rsi_ok = (rsi > {params['rsi_entry']}) & (rsi < {params['rsi_exit']})
    return xr.where(trend_ok & rsi_ok, 1, 0) * is_liquid"""

    elif alpha_name == "breakout_long":
        strategy_body = f"""
    high = data.sel(field="high")
    upper = high.rolling(time={params['channel']}).max()
    ema_filter = qnta.ema(close, {params['ema_filter']})
    breakout = (close >= upper) & (close > ema_filter)
    return xr.where(breakout, 1, 0) * is_liquid"""

    elif alpha_name == "vol_weighted":
        strategy_body = f"""
    returns = close / close.shift(time=1) - 1
    vol = returns.rolling(time={params['vol_window']}).std()
    inv_vol = (1.0 / vol.where(vol > 1e-10)).fillna(0)
    ema = qnta.ema(close, {params['trend_ema']})
    trend = xr.where(close > ema, 1, 0)
    return inv_vol * trend * is_liquid"""

    elif alpha_name == "macd_long":
        strategy_body = f"""
    macd_line, signal_line, histogram = qnta.macd(close, {params['fast']}, {params['slow']}, {params['signal']})
    return xr.where(histogram > 0, 1, 0) * is_liquid"""

    elif alpha_name == "combined_score":
        strategy_body = f"""
    ema = qnta.ema(close, {params['ema_period']})
    trend_score = xr.where(close > ema, 1, 0)
    mom = close / close.shift(time={params['mom_period']}) - 1
    mom_score = xr.where(mom > 0, 1, 0)
    rsi = qnta.rsi(close, {params['rsi_period']})
    rsi_score = xr.where(rsi < {params['rsi_cap']}, 1, 0)
    total = trend_score + mom_score + rsi_score
    return xr.where(total >= {params['min_score']}, 1, 0) * is_liquid"""

    else:
        raise ValueError(f"Unknown alpha: {alpha_name}")

    code = f'''# Auto-generated by Q24 Alpha Optimizer
# Alpha: {alpha_name}
# Params: {json.dumps(params)}
# Generated: {datetime.now().isoformat()}

import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import numpy as np
import qnt.stats as qnstats
import qnt.data as qndata
import qnt.output as qnout
import qnt.ta as qnta


def load_data(period):
    return qndata.cryptodaily_load_data(tail=period)


def strategy(data):
    close = data.sel(field="close")
    is_liquid = data.sel(field="is_liquid")
{strategy_body}


# ── Single-pass backtest ──
data = qndata.cryptodaily_load_data(min_date="2015-01-01")
weights = strategy(data)
weights = qnout.clean(weights, data, "crypto_daily_long")

stats = qnstats.calc_stat(data, weights.sel(time=slice("2016-01-01", None))).sel(time=slice("2016-01-01", None))
print(stats.to_pandas().tail(1))

qnout.write(weights)
'''
    return code


def create_submission_notebook(code: str, alpha_name: str, params: Dict, metrics: Dict) -> dict:
    """Create strategy.ipynb for Quantiacs submission."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": [
            {
                "cell_type": "markdown", "metadata": {}, "id": "header",
                "source": [
                    f"# Q24 Crypto — {alpha_name}\n",
                    f"**Params**: `{json.dumps(params)}`\n\n",
                    f"**Sharpe**: {metrics.get('sharpe', 'N/A'):.4f} | ",
                    f"**Return**: {metrics.get('return', 0):.4f} | ",
                    f"**MaxDD**: {metrics.get('max_drawdown', 0):.4f}\n",
                ],
            },
            {
                "cell_type": "code", "metadata": {}, "id": "strategy",
                "source": code.split("\n"),
                "execution_count": None, "outputs": [],
            },
        ],
    }


def export_submission(alpha_name: str, params: Dict, metrics: Dict,
                      output_dir: str = "submissions") -> str:
    """Export a complete submission package."""
    out = Path(output_dir)
    # Create unique folder name
    param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    folder_name = f"{alpha_name}_{param_hash}"
    sub_dir = out / folder_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    code = generate_strategy_code(alpha_name, params)

    # Write strategy.py
    (sub_dir / "strategy.py").write_text(code)

    # Write strategy.ipynb
    nb = create_submission_notebook(code, alpha_name, params, metrics)
    (sub_dir / "strategy.ipynb").write_text(json.dumps(nb, indent=2))

    # Write init.ipynb (empty)
    init = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [{"cell_type": "code", "metadata": {}, "id": "init",
                    "source": ["pass\n"], "execution_count": None, "outputs": []}],
    }
    (sub_dir / "init.ipynb").write_text(json.dumps(init, indent=2))

    # Write metadata
    meta = {"alpha": alpha_name, "params": params, "metrics": metrics,
            "generated": datetime.now().isoformat()}
    (sub_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return str(sub_dir)


# ═══════════════════════════════════════════════════════════════════
#  RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════════

def print_results(study: optuna.Study, top_k: int = 10):
    """Print optimization results."""
    print("\n")
    print("╔" + "═" * 80 + "╗")
    print("║" + "  Q24 CRYPTO ALPHA OPTIMIZER — RESULTS".center(80) + "║")
    print("╠" + "═" * 80 + "╣")

    trials = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)
    valid = [t for t in trials if t.value and t.value > -999]

    passed = sum(1 for t in valid if t.value >= 1.0)
    print(f"║  Trials: {len(study.trials)} | Valid: {len(valid)} | "
          f"Sharpe≥1.0: {passed} | Best: {study.best_value:.4f}".ljust(80) + "║")
    print("╠" + "═" * 80 + "╣")

    header = f"  {'#':>3} {'Alpha':<20} {'Sharpe':>8} {'Return':>8} {'MaxDD':>8} {'Params':<25}"
    print("║" + header.ljust(80) + "║")
    print("║" + "─" * 80 + "║")

    for i, t in enumerate(valid[:top_k], 1):
        m = t.user_attrs.get("metrics", {})
        aname = t.user_attrs.get("alpha_name", "?")
        icon = "✅" if t.value >= 1.0 else "⚠️"
        params_str = str(t.user_attrs.get("params", {}))
        if len(params_str) > 23:
            params_str = params_str[:20] + "..."
        line = (f"  {i:>3} {aname:<20} {t.value:>8.4f} "
                f"{m.get('return', 0):>7.4f} {m.get('max_drawdown', 0):>8.4f} {params_str:<25}")
        print("║" + f" {icon} " + line[:77].ljust(77) + "║")

    print("╚" + "═" * 80 + "╝")


# ═══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_optimization(n_trials: int = 100, top_k_export: int = 3,
                     output_dir: str = "submissions",
                     study_name: str = "q24_crypto") -> optuna.Study:
    """Run full Optuna optimization pipeline."""
    print("\n" + "🚀 " * 20)
    print("  Q24 CRYPTO ALPHA OPTIMIZER")
    print("🚀 " * 20)

    # Load data once
    data = get_data()

    # Create study
    db_path = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=db_path,
        load_if_exists=True,
    )

    # Run optimization
    print(f"\n⚡ Running {n_trials} trials (Optuna TPE sampler)...")
    print(f"   Database: {study_name}.db (resume-safe)\n")

    objective = create_objective(data)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Print results
    print_results(study, top_k=10)

    # Export top-K
    trials = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)
    valid = [t for t in trials if t.value and t.value >= 1.0]

    if valid:
        print(f"\n📦 Exporting top-{min(top_k_export, len(valid))} strategies...")
        for i, t in enumerate(valid[:top_k_export]):
            aname = t.user_attrs.get("alpha_name", "unknown")
            params = t.user_attrs.get("params", {})
            metrics = t.user_attrs.get("metrics", {})
            sub_path = export_submission(aname, params, metrics, output_dir)
            print(f"   [{i+1}] {aname} (Sharpe={t.value:.4f}) → {sub_path}")

        print(f"\n✅ Upload strategy.ipynb files from {output_dir}/ to quantiacs.com")
    else:
        print("\n⚠️  No strategies achieved Sharpe ≥ 1.0 yet.")
        print("   Try more trials: python q24_optimizer.py --n-trials 300")

    return study


def export_trial(study_name: str, trial_number: int, output_dir: str = "submissions"):
    """Export a specific trial as submission."""
    db_path = f"sqlite:///{study_name}.db"
    study = optuna.load_study(study_name=study_name, storage=db_path)

    t = study.trials[trial_number]
    aname = t.user_attrs.get("alpha_name", "unknown")
    params = t.user_attrs.get("params", {})
    metrics = t.user_attrs.get("metrics", {})

    sub_path = export_submission(aname, params, metrics, output_dir)
    print(f"📦 Exported trial #{trial_number}: {aname} (Sharpe={t.value:.4f})")
    print(f"   → {sub_path}")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Q24 Crypto Alpha Optimizer")
    parser.add_argument("--mode", default="optimize",
                        choices=["optimize", "export", "results"],
                        help="Mode: optimize, export, or results")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of Optuna trials (default: 100)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Top-K strategies to export (default: 3)")
    parser.add_argument("--trial", type=int, default=0,
                        help="Trial number to export (for --mode export)")
    parser.add_argument("--study-name", default="q24_crypto",
                        help="Optuna study name (default: q24_crypto)")
    parser.add_argument("--output-dir", default="submissions",
                        help="Output directory (default: submissions)")

    args = parser.parse_args()

    if args.mode == "optimize":
        run_optimization(
            n_trials=args.n_trials,
            top_k_export=args.top_k,
            output_dir=args.output_dir,
            study_name=args.study_name,
        )
    elif args.mode == "export":
        export_trial(args.study_name, args.trial, args.output_dir)
    elif args.mode == "results":
        db_path = f"sqlite:///{args.study_name}.db"
        study = optuna.load_study(study_name=args.study_name, storage=db_path)
        print_results(study, top_k=20)


if __name__ == "__main__":
    main()
