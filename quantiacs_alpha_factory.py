#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║              QUANTIACS ALPHA FACTORY v1.0                          ║
║  Auto-generate, backtest & submit alpha strategies to Quantiacs    ║
║  Author: Built for Thanh @ Talyxion LLC                            ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
    python quantiacs_alpha_factory.py --mode generate --competition futures --num-alphas 20
    python quantiacs_alpha_factory.py --mode backtest --strategy strategy_sma_cross.py
    python quantiacs_alpha_factory.py --mode pipeline --competition futures --num-alphas 50 --top-k 5
    python quantiacs_alpha_factory.py --mode submit --strategy strategy_sma_cross.py --api-key YOUR_KEY

Competition types: futures, stocks, cryptodaily, cryptofutures
"""

import os
import sys
import json
import time
import shutil
import hashlib
import argparse
import itertools
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import textwrap
import traceback

# ═══════════════════════════════════════════════════════════════════
#  ALPHA TEMPLATES REGISTRY
# ═══════════════════════════════════════════════════════════════════

ALPHA_TEMPLATES: Dict[str, dict] = {}


def register_alpha(name: str, category: str, description: str, 
                   competition_types: List[str], param_grid: Dict[str, list]):
    """Decorator to register an alpha template."""
    def decorator(func):
        ALPHA_TEMPLATES[name] = {
            "name": name,
            "category": category,
            "description": description,
            "competition_types": competition_types,
            "param_grid": param_grid,
            "generator": func,
        }
        return func
    return decorator


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY 1: TREND FOLLOWING ALPHAS
# ═══════════════════════════════════════════════════════════════════

@register_alpha(
    name="sma_crossover",
    category="trend",
    description="Dual SMA crossover long/short",
    competition_types=["futures", "stocks", "cryptodaily", "cryptofutures"],
    param_grid={
        "fast_period": [5, 10, 15, 20, 30],
        "slow_period": [50, 100, 150, 200],
    }
)
def gen_sma_crossover(params, competition_type):
    fast = params["fast_period"]
    slow = params["slow_period"]
    if fast >= slow:
        return None
    return f'''
import qnt.ta as qnta
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    sma_fast = qnta.sma(close, {fast}).isel(time=-1)
    sma_slow = qnta.sma(close, {slow}).isel(time=-1)
    weights = xr.where(sma_fast > sma_slow, 1, -1)
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={slow + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


@register_alpha(
    name="ema_crossover",
    category="trend",
    description="Dual EMA crossover with momentum filter",
    competition_types=["futures", "stocks", "cryptodaily", "cryptofutures"],
    param_grid={
        "fast_period": [8, 12, 21],
        "slow_period": [50, 100, 200],
    }
)
def gen_ema_crossover(params, competition_type):
    fast = params["fast_period"]
    slow = params["slow_period"]
    if fast >= slow:
        return None
    return f'''
import qnt.ta as qnta
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    ema_fast = qnta.ema(close, {fast}).isel(time=-1)
    ema_slow = qnta.ema(close, {slow}).isel(time=-1)
    weights = xr.where(ema_fast > ema_slow, 1, -1)
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={slow + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


@register_alpha(
    name="triple_ema",
    category="trend",
    description="Triple EMA trend system (fast/medium/slow alignment)",
    competition_types=["futures", "stocks", "cryptodaily"],
    param_grid={
        "fast": [5, 8, 13],
        "medium": [21, 34, 55],
        "slow": [89, 144, 200],
    }
)
def gen_triple_ema(params, competition_type):
    f, m, s = params["fast"], params["medium"], params["slow"]
    if not (f < m < s):
        return None
    return f'''
import qnt.ta as qnta
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    ema_f = qnta.ema(close, {f}).isel(time=-1)
    ema_m = qnta.ema(close, {m}).isel(time=-1)
    ema_s = qnta.ema(close, {s}).isel(time=-1)
    bullish = (ema_f > ema_m) & (ema_m > ema_s)
    bearish = (ema_f < ema_m) & (ema_m < ema_s)
    weights = xr.where(bullish, 1, xr.where(bearish, -1, 0))
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={s + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


@register_alpha(
    name="trix_signal",
    category="trend",
    description="TRIX indicator trend-following",
    competition_types=["futures", "stocks", "cryptodaily"],
    param_grid={
        "trix_period": [12, 15, 20, 30],
        "signal_period": [5, 9, 14],
    }
)
def gen_trix_signal(params, competition_type):
    tp = params["trix_period"]
    sp = params["signal_period"]
    return f'''
import qnt.ta as qnta
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    trix_val = qnta.trix(close, {tp}).isel(time=-1)
    weights = xr.where(trix_val > 0, 1, -1)
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={tp * 4 + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY 2: MEAN REVERSION ALPHAS
# ═══════════════════════════════════════════════════════════════════

@register_alpha(
    name="bollinger_reversion",
    category="mean_reversion",
    description="Bollinger Band mean reversion strategy",
    competition_types=["futures", "stocks", "cryptodaily"],
    param_grid={
        "bb_period": [15, 20, 30, 40],
        "bb_std": [1.5, 2.0, 2.5, 3.0],
    }
)
def gen_bollinger_reversion(params, competition_type):
    period = params["bb_period"]
    std_dev = params["bb_std"]
    return f'''
import qnt.ta as qnta
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr
import numpy as np

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    sma = qnta.sma(close, {period}).isel(time=-1)
    std = close.rolling(time={period}).std().isel(time=-1)
    upper = sma + {std_dev} * std
    lower = sma - {std_dev} * std
    cur_price = close.isel(time=-1)
    weights = xr.where(cur_price < lower, 1, xr.where(cur_price > upper, -1, 0))
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={period + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


@register_alpha(
    name="rsi_reversion",
    category="mean_reversion",
    description="RSI-based mean reversion",
    competition_types=["futures", "stocks", "cryptodaily", "cryptofutures"],
    param_grid={
        "rsi_period": [7, 14, 21],
        "oversold": [20, 25, 30],
        "overbought": [70, 75, 80],
    }
)
def gen_rsi_reversion(params, competition_type):
    rp = params["rsi_period"]
    os_level = params["oversold"]
    ob_level = params["overbought"]
    if os_level >= ob_level:
        return None
    return f'''
import qnt.ta as qnta
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    rsi = qnta.rsi(close, {rp}).isel(time=-1)
    weights = xr.where(rsi < {os_level}, 1, xr.where(rsi > {ob_level}, -1, 0))
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={rp + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


@register_alpha(
    name="zscore_reversion",
    category="mean_reversion",
    description="Z-Score based mean reversion with rolling window",
    competition_types=["futures", "stocks", "cryptodaily"],
    param_grid={
        "lookback": [20, 40, 60, 90],
        "entry_z": [1.5, 2.0, 2.5],
    }
)
def gen_zscore_reversion(params, competition_type):
    lb = params["lookback"]
    ez = params["entry_z"]
    return f'''
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    mean = close.rolling(time={lb}).mean().isel(time=-1)
    std = close.rolling(time={lb}).std().isel(time=-1)
    cur = close.isel(time=-1)
    zscore = (cur - mean) / std.where(std > 0)
    weights = xr.where(zscore < -{ez}, 1, xr.where(zscore > {ez}, -1, 0))
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={lb + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY 3: MOMENTUM / CROSS-SECTIONAL ALPHAS
# ═══════════════════════════════════════════════════════════════════

@register_alpha(
    name="momentum_rank",
    category="momentum",
    description="Cross-sectional momentum ranking",
    competition_types=["futures", "stocks", "cryptodaily"],
    param_grid={
        "momentum_period": [20, 40, 60, 120, 252],
        "top_n_pct": [0.2, 0.3, 0.4],
    }
)
def gen_momentum_rank(params, competition_type):
    mp = params["momentum_period"]
    tn = params["top_n_pct"]
    return f'''
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    returns = (close.isel(time=-1) / close.isel(time=-{mp}) - 1)
    rank = returns.rank("asset")
    n_assets = rank.count("asset") if hasattr(rank, "count") else len(rank.asset)
    top_threshold = rank.max("asset") * (1 - {tn})
    bot_threshold = rank.max("asset") * {tn}
    weights = xr.where(rank >= top_threshold, 1, xr.where(rank <= bot_threshold, -1, 0))
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={mp + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


@register_alpha(
    name="rate_of_change",
    category="momentum",
    description="Rate of Change (ROC) momentum strategy",
    competition_types=["futures", "stocks", "cryptodaily", "cryptofutures"],
    param_grid={
        "roc_period": [10, 20, 40, 60],
    }
)
def gen_rate_of_change(params, competition_type):
    rp = params["roc_period"]
    return f'''
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    roc = (close.isel(time=-1) / close.isel(time=-{rp}) - 1)
    weights = xr.where(roc > 0, 1, -1)
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={rp + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY 4: VOLATILITY ALPHAS
# ═══════════════════════════════════════════════════════════════════

@register_alpha(
    name="atr_breakout",
    category="volatility",
    description="ATR-based breakout strategy",
    competition_types=["futures", "stocks", "cryptodaily"],
    param_grid={
        "atr_period": [10, 14, 20],
        "atr_mult": [1.5, 2.0, 2.5, 3.0],
        "ma_period": [20, 50],
    }
)
def gen_atr_breakout(params, competition_type):
    ap = params["atr_period"]
    am = params["atr_mult"]
    mp = params["ma_period"]
    return f'''
import qnt.ta as qnta
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    high = data.sel(field="high")
    low = data.sel(field="low")
    close = data.sel(field="close")
    atr = qnta.atr(high, low, close, {ap}).isel(time=-1)
    ma = qnta.sma(close, {mp}).isel(time=-1)
    cur = close.isel(time=-1)
    weights = xr.where(cur > ma + {am} * atr, 1, xr.where(cur < ma - {am} * atr, -1, 0))
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={max(ap, mp) + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


@register_alpha(
    name="vol_adjusted_momentum",
    category="volatility",
    description="Volatility-adjusted momentum (risk parity style)",
    competition_types=["futures", "stocks", "cryptodaily"],
    param_grid={
        "mom_period": [20, 60, 120],
        "vol_period": [20, 40, 60],
    }
)
def gen_vol_adjusted_momentum(params, competition_type):
    mp = params["mom_period"]
    vp = params["vol_period"]
    return f'''
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr
import numpy as np

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    returns = close / close.shift(time=1) - 1
    momentum = (close.isel(time=-1) / close.isel(time=-{mp}) - 1)
    volatility = returns.isel(time=slice(-{vp}, None)).std("time")
    vol_safe = volatility.where(volatility > 1e-10).fillna(1)
    weights = momentum / vol_safe
    abs_sum = abs(weights).sum("asset")
    weights = weights / abs_sum.where(abs_sum > 0).fillna(1)
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={max(mp, vp) + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY 5: MULTI-INDICATOR COMBO ALPHAS
# ═══════════════════════════════════════════════════════════════════

@register_alpha(
    name="macd_rsi_combo",
    category="combo",
    description="MACD + RSI combined signal",
    competition_types=["futures", "stocks", "cryptodaily"],
    param_grid={
        "macd_fast": [8, 12],
        "macd_slow": [21, 26],
        "macd_signal": [5, 9],
        "rsi_period": [14, 21],
    }
)
def gen_macd_rsi_combo(params, competition_type):
    mf = params["macd_fast"]
    ms = params["macd_slow"]
    msig = params["macd_signal"]
    rp = params["rsi_period"]
    if mf >= ms:
        return None
    return f'''
import qnt.ta as qnta
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    macd_line, macd_sig, macd_hist = qnta.macd(close, {mf}, {ms}, {msig})
    macd_hist_now = macd_hist.isel(time=-1)
    rsi = qnta.rsi(close, {rp}).isel(time=-1)
    bull = (macd_hist_now > 0) & (rsi > 50)
    bear = (macd_hist_now < 0) & (rsi < 50)
    weights = xr.where(bull, 1, xr.where(bear, -1, 0))
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={ms + msig + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


@register_alpha(
    name="ema_rsi_atr_combo",
    category="combo",
    description="EMA trend + RSI filter + ATR volatility sizing",
    competition_types=["futures", "stocks"],
    param_grid={
        "ema_period": [50, 100, 200],
        "rsi_period": [14],
        "atr_period": [14, 20],
    }
)
def gen_ema_rsi_atr_combo(params, competition_type):
    ep = params["ema_period"]
    rp = params["rsi_period"]
    ap = params["atr_period"]
    return f'''
import qnt.ta as qnta
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    high = data.sel(field="high")
    low = data.sel(field="low")
    close = data.sel(field="close")
    ema = qnta.ema(close, {ep}).isel(time=-1)
    rsi = qnta.rsi(close, {rp}).isel(time=-1)
    atr = qnta.atr(high, low, close, {ap}).isel(time=-1)
    cur = close.isel(time=-1)
    inv_vol = 1.0 / atr.where(atr > 1e-10).fillna(1)
    trend_up = (cur > ema) & (rsi > 40) & (rsi < 80)
    trend_dn = (cur < ema) & (rsi > 20) & (rsi < 60)
    weights = xr.where(trend_up, inv_vol, xr.where(trend_dn, -inv_vol, 0))
    abs_sum = abs(weights).sum("asset")
    weights = weights / abs_sum.where(abs_sum > 0).fillna(1)
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={max(ep, rp, ap) + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY 6: LONG-ONLY ALPHAS (for crypto long-only contests)
# ═══════════════════════════════════════════════════════════════════

@register_alpha(
    name="momentum_long_only",
    category="long_only",
    description="Momentum-weighted long-only allocation",
    competition_types=["cryptodaily"],
    param_grid={
        "lookback": [7, 14, 30, 60],
        "min_momentum": [0.0, 0.01, 0.02],
    }
)
def gen_momentum_long_only(params, competition_type):
    lb = params["lookback"]
    mm = params["min_momentum"]
    return f'''
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    is_liquid = data.sel(field="is_liquid").isel(time=-1)
    mom = (close.isel(time=-1) / close.isel(time=-{lb}) - 1)
    mom_positive = mom.where(mom > {mm}).fillna(0)
    weights = mom_positive * is_liquid
    total = weights.sum("asset")
    weights = weights / total.where(total > 0).fillna(1)
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={lb + 30},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


@register_alpha(
    name="equal_weight_liquid",
    category="long_only",
    description="Equal-weight allocation to all liquid assets",
    competition_types=["cryptodaily"],
    param_grid={
        "sma_filter_period": [0, 20, 50],
    }
)
def gen_equal_weight_liquid(params, competition_type):
    sfp = params["sma_filter_period"]
    sma_block = ""
    if sfp > 0:
        sma_block = f"""
    sma = qnta.sma(close, {sfp}).isel(time=-1)
    cur = close.isel(time=-1)
    trend_filter = xr.where(cur > sma, 1, 0)
    weights = weights * trend_filter"""
    
    imports = "import qnt.ta as qnta\n" if sfp > 0 else ""
    return f'''
{imports}import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    is_liquid = data.sel(field="is_liquid").isel(time=-1)
    weights = is_liquid * 1.0{sma_block}
    total = weights.sum("asset")
    weights = weights / total.where(total > 0).fillna(1)
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={max(sfp, 10) + 30},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY 7: PRICE-ACTION / PATTERN ALPHAS
# ═══════════════════════════════════════════════════════════════════

@register_alpha(
    name="channel_breakout",
    category="breakout",
    description="Donchian Channel breakout system",
    competition_types=["futures", "stocks", "cryptodaily"],
    param_grid={
        "channel_period": [10, 20, 40, 55],
    }
)
def gen_channel_breakout(params, competition_type):
    cp = params["channel_period"]
    return f'''
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    high = data.sel(field="high")
    low = data.sel(field="low")
    close = data.sel(field="close")
    upper = high.rolling(time={cp}).max().isel(time=-1)
    lower = low.rolling(time={cp}).min().isel(time=-1)
    cur = close.isel(time=-1)
    weights = xr.where(cur >= upper, 1, xr.where(cur <= lower, -1, 0))
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={cp + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


@register_alpha(
    name="range_compression",
    category="breakout",
    description="Range compression → expansion breakout",
    competition_types=["futures", "stocks"],
    param_grid={
        "short_window": [5, 10],
        "long_window": [20, 40, 60],
        "compression_ratio": [0.3, 0.5, 0.7],
    }
)
def gen_range_compression(params, competition_type):
    sw = params["short_window"]
    lw = params["long_window"]
    cr = params["compression_ratio"]
    if sw >= lw:
        return None
    return f'''
import qnt.data as qndata
import qnt.ta as qnta
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    high = data.sel(field="high")
    low = data.sel(field="low")
    range_short = (high.rolling(time={sw}).max() - low.rolling(time={sw}).min()).isel(time=-1)
    range_long = (high.rolling(time={lw}).max() - low.rolling(time={lw}).min()).isel(time=-1)
    compressed = range_short < {cr} * range_long
    ema = qnta.ema(close, {sw}).isel(time=-1)
    cur = close.isel(time=-1)
    weights = xr.where(compressed & (cur > ema), 1, xr.where(compressed & (cur < ema), -1, 0))
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={lw + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _data_loader(competition_type: str) -> str:
    """Return the correct qndata loader function name."""
    return {
        "futures": "futures_load_data",
        "stocks": "stocks.load_ndx_data",
        "cryptodaily": "cryptodaily.load_data",
        "cryptofutures": "cryptofutures.load_data",
    }.get(competition_type, "futures_load_data")


def _liquidity_filter(competition_type: str) -> str:
    """Return liquidity filter code if needed."""
    if competition_type in ("cryptodaily",):
        return 'weights = weights * data.sel(field="is_liquid").isel(time=-1)'
    return ""


# ═══════════════════════════════════════════════════════════════════
#  BACKTEST RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    strategy_name: str
    template_name: str
    params: Dict[str, Any]
    competition_type: str
    sharpe_ratio: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    status: str = "pending"  # pending, success, failed, filtered
    error_message: str = ""
    file_path: str = ""
    backtest_log: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def passes_filter(self) -> bool:
        return self.sharpe_ratio >= 0.7 and self.status == "success"

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════
#  ALPHA GENERATOR ENGINE
# ═══════════════════════════════════════════════════════════════════

class AlphaGenerator:
    """Generates alpha strategy code from templates and parameter grids."""

    def __init__(self, competition_type: str = "futures", output_dir: str = "generated_alphas",
                 api_key: str = ""):
        self.competition_type = competition_type
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_compatible_templates(self) -> List[str]:
        """Return template names compatible with the competition type."""
        return [
            name for name, tpl in ALPHA_TEMPLATES.items()
            if self.competition_type in tpl["competition_types"]
        ]

    def generate_param_combinations(self, template_name: str) -> List[Dict]:
        """Generate all parameter combinations for a template."""
        tpl = ALPHA_TEMPLATES[template_name]
        grid = tpl["param_grid"]
        keys = list(grid.keys())
        values = list(grid.values())
        combos = []
        for combo in itertools.product(*values):
            combos.append(dict(zip(keys, combo)))
        return combos

    def generate_strategy(self, template_name: str, params: Dict) -> Optional[Tuple[str, str]]:
        """Generate a strategy .py file. Returns (filepath, code) or None."""
        tpl = ALPHA_TEMPLATES[template_name]
        code = tpl["generator"](params, self.competition_type)
        if code is None:
            return None

        # Create unique name
        param_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
        name = f"strategy_{template_name}_{param_str}".replace(".", "p")
        filename = f"{name}.py"
        filepath = self.output_dir / filename

        # Add header
        api_line = f'os.environ["API_KEY"] = "{self.api_key}"' if self.api_key else '# os.environ["API_KEY"] = "YOUR_API_KEY_HERE"'
        header = f'''# Auto-generated by Quantiacs Alpha Factory
# Template: {template_name}
# Competition: {self.competition_type}
# Params: {json.dumps(params)}
# Generated: {datetime.now().isoformat()}
import os
{api_line}
'''
        full_code = header + code.strip() + "\n"
        filepath.write_text(full_code)
        return str(filepath), full_code

    def generate_batch(self, num_alphas: int = 20, 
                       templates: Optional[List[str]] = None) -> List[Tuple[str, str, Dict]]:
        """Generate a batch of alphas. Returns list of (filepath, template_name, params)."""
        if templates is None:
            templates = self.get_compatible_templates()

        all_candidates = []
        for tname in templates:
            combos = self.generate_param_combinations(tname)
            for combo in combos:
                all_candidates.append((tname, combo))

        # Limit to num_alphas, spread across templates
        import random
        random.shuffle(all_candidates)
        selected = all_candidates[:num_alphas]

        results = []
        for tname, params in selected:
            result = self.generate_strategy(tname, params)
            if result:
                filepath, code = result
                results.append((filepath, tname, params))

        return results


# ═══════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════

class BacktestEngine:
    """Run backtests on generated strategies and parse results."""

    def __init__(self, timeout: int = 600, api_key: str = ""):
        self.timeout = timeout
        self.api_key = api_key

    def run_backtest(self, strategy_path: str, template_name: str = "",
                     params: Dict = None) -> BacktestResult:
        """Run a single backtest by executing the strategy file."""
        result = BacktestResult(
            strategy_name=Path(strategy_path).stem,
            template_name=template_name,
            params=params or {},
            competition_type="",
            file_path=strategy_path,
        )

        # Read competition type from file
        try:
            with open(strategy_path, "r") as f:
                code = f.read()
            for line in code.split("\n"):
                if "competition_type=" in line and '"' in line:
                    ct = line.split('"')[1] if '"' in line else ""
                    if ct:
                        result.competition_type = ct
                        break
        except Exception as e:
            result.status = "failed"
            result.error_message = f"Cannot read file: {e}"
            return result

        # Execute backtest
        try:
            # Pass API_KEY to subprocess
            env = os.environ.copy()
            if self.api_key:
                env["API_KEY"] = self.api_key
            abs_strategy_path = str(Path(strategy_path).resolve())
            proc = subprocess.run(
                [sys.executable, abs_strategy_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            output = proc.stdout + proc.stderr
            result.backtest_log = output[-5000:]  # Keep last 5000 chars

            if proc.returncode != 0:
                result.status = "failed"
                # Detect common errors for clearer messages
                if "HTTP Error 500" in output:
                    result.error_message = "Quantiacs data server returned HTTP 500. Server may be temporarily down. Try again later."
                elif "Please, specify the API_KEY" in output:
                    result.error_message = "API_KEY not set. Run: $env:API_KEY = 'your_key' or pass --api-key"
                elif "can't download" in output:
                    result.error_message = "Data download failed. Check API key and internet connection."
                elif "No such file" in output:
                    result.error_message = f"File not found. Path issue in subprocess."
                else:
                    result.error_message = f"Exit code {proc.returncode}: {output[-500:]}"
                return result

            # Parse metrics from output
            result = self._parse_metrics(result, output)
            
        except subprocess.TimeoutExpired:
            result.status = "failed"
            result.error_message = f"Timeout after {self.timeout}s"
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)

        return result

    def _parse_metrics(self, result: BacktestResult, output: str) -> BacktestResult:
        """Parse Sharpe ratio and other metrics from backtest output."""
        import re

        result.status = "success"

        # Parse Sharpe ratio
        sharpe_patterns = [
            r"Sharpe\s*(?:ratio|Ratio)?\s*[:=]\s*([-\d.]+)",
            r"sharpe\s*[:=]\s*([-\d.]+)",
            r"In-Sample Sharpe\s*[:=]\s*([-\d.]+)",
        ]
        for pat in sharpe_patterns:
            match = re.search(pat, output, re.IGNORECASE)
            if match:
                try:
                    result.sharpe_ratio = float(match.group(1))
                except ValueError:
                    pass
                break

        # Parse annual return
        ret_patterns = [
            r"(?:Annual|Annualized)\s*(?:Return|return)\s*[:=]\s*([-\d.]+)",
            r"mean_return\s*[:=]\s*([-\d.]+)",
        ]
        for pat in ret_patterns:
            match = re.search(pat, output, re.IGNORECASE)
            if match:
                try:
                    result.annual_return = float(match.group(1))
                except ValueError:
                    pass
                break

        # Parse max drawdown
        dd_patterns = [
            r"(?:Max|Maximum)\s*(?:Drawdown|drawdown)\s*[:=]\s*([-\d.]+)",
            r"max_drawdown\s*[:=]\s*([-\d.]+)",
        ]
        for pat in dd_patterns:
            match = re.search(pat, output, re.IGNORECASE)
            if match:
                try:
                    result.max_drawdown = float(match.group(1))
                except ValueError:
                    pass
                break

        if result.sharpe_ratio < 0.7:
            result.status = "filtered"

        return result

    def run_batch(self, strategies: List[Tuple[str, str, Dict]], 
                  max_workers: int = 1) -> List[BacktestResult]:
        """Run backtests on a batch of strategies."""
        results = []
        total = len(strategies)
        for i, (filepath, tname, params) in enumerate(strategies):
            print(f"\n{'='*60}")
            print(f"  [{i+1}/{total}] Backtesting: {Path(filepath).stem}")
            print(f"  Template: {tname} | Params: {params}")
            print(f"{'='*60}")
            result = self.run_backtest(filepath, tname, params)
            results.append(result)
            status_icon = "✅" if result.passes_filter else ("❌" if result.status == "failed" else "⚠️")
            print(f"  {status_icon} Sharpe: {result.sharpe_ratio:.4f} | Status: {result.status}")
        return results


# ═══════════════════════════════════════════════════════════════════
#  SUBMISSION MANAGER
# ═══════════════════════════════════════════════════════════════════

class SubmissionManager:
    """Prepare and manage strategy submissions to Quantiacs."""

    def __init__(self, api_key: str = "", output_dir: str = "submissions"):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_submission(self, strategy_path: str, 
                           strategy_name: str = "") -> str:
        """Prepare a strategy for Quantiacs submission."""
        if not strategy_name:
            strategy_name = Path(strategy_path).stem

        sub_dir = self.output_dir / strategy_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        # Read strategy code
        with open(strategy_path, "r") as f:
            code = f.read()

        # For submission to Quantiacs platform, comment out the API key
        # (the platform injects its own key server-side)
        import re
        code = re.sub(
            r'^os\.environ\["API_KEY"\]\s*=\s*"[^"]*"',
            '# os.environ["API_KEY"] = "YOUR_API_KEY_HERE"  # Set by Quantiacs platform',
            code,
            flags=re.MULTILINE,
        )

        # Write strategy.py
        strategy_file = sub_dir / "strategy.py"
        strategy_file.write_text(code)

        # Create strategy.ipynb (Jupyter notebook for Quantiacs platform)
        notebook = self._create_notebook(code, strategy_name)
        notebook_file = sub_dir / "strategy.ipynb"
        notebook_file.write_text(json.dumps(notebook, indent=2))

        # Create init.ipynb for dependencies
        init_nb = self._create_init_notebook()
        init_file = sub_dir / "init.ipynb"
        init_file.write_text(json.dumps(init_nb, indent=2))

        # Create precheck.ipynb
        precheck_nb = self._create_precheck_notebook()
        precheck_file = sub_dir / "precheck.ipynb"
        precheck_file.write_text(json.dumps(precheck_nb, indent=2))

        print(f"📦 Submission prepared at: {sub_dir}")
        print(f"   - strategy.ipynb (upload to Quantiacs Jupyter)")
        print(f"   - strategy.py (local reference)")
        print(f"   - init.ipynb (dependencies)")
        print(f"   - precheck.ipynb (pre-submission check)")
        return str(sub_dir)

    def _create_notebook(self, code: str, name: str) -> dict:
        """Create a Jupyter notebook from strategy code."""
        cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {name}\n",
                    f"Auto-generated by Quantiacs Alpha Factory\n",
                    f"Generated: {datetime.now().isoformat()}\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": code.split("\n"),
                "execution_count": None,
                "outputs": [],
            },
        ]
        return {
            "nbformat": 4,
            "nbformat_minor": 4,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.10.0"},
            },
            "cells": cells,
        }

    def _create_init_notebook(self) -> dict:
        """Create init.ipynb for external dependencies."""
        return {
            "nbformat": 4, "nbformat_minor": 4,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [{
                "cell_type": "code", "metadata": {},
                "source": ["# No external dependencies needed\n", "pass\n"],
                "execution_count": None, "outputs": [],
            }],
        }

    def _create_precheck_notebook(self) -> dict:
        """Create precheck.ipynb."""
        return {
            "nbformat": 4, "nbformat_minor": 4,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [{
                "cell_type": "code", "metadata": {},
                "source": [
                    "import qnt.output as qnout\n",
                    "# Run precheck\n",
                    "# qnout.check(weights, data, competition_type)\n",
                ],
                "execution_count": None, "outputs": [],
            }],
        }


# ═══════════════════════════════════════════════════════════════════
#  RESULTS DASHBOARD
# ═══════════════════════════════════════════════════════════════════

class ResultsDashboard:
    """Generate summary reports and rankings."""

    @staticmethod
    def print_summary(results: List[BacktestResult]):
        """Print a formatted summary of backtest results."""
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + "  QUANTIACS ALPHA FACTORY — BACKTEST RESULTS".center(78) + "║")
        print("╠" + "═" * 78 + "╣")

        # Stats
        total = len(results)
        passed = sum(1 for r in results if r.passes_filter)
        failed = sum(1 for r in results if r.status == "failed")
        filtered = sum(1 for r in results if r.status == "filtered")
        
        print(f"║  Total: {total} | Passed (SR≥0.7): {passed} | "
              f"Filtered: {filtered} | Failed: {failed}".ljust(78) + "║")
        print("╠" + "═" * 78 + "╣")

        # Sort by Sharpe
        sorted_results = sorted(results, key=lambda r: r.sharpe_ratio, reverse=True)

        # Header
        header = f"  {'#':>3} {'Strategy':<35} {'Sharpe':>8} {'Return':>8} {'MaxDD':>8} {'Status':>8}"
        print("║" + header.ljust(78) + "║")
        print("║" + "─" * 78 + "║")

        for i, r in enumerate(sorted_results[:30], 1):
            icon = "✅" if r.passes_filter else ("❌" if r.status == "failed" else "⚠️")
            name = r.strategy_name[:33]
            line = f"  {i:>3} {name:<35} {r.sharpe_ratio:>8.3f} {r.annual_return:>7.1f}% {r.max_drawdown:>7.1f}% {icon:>5}"
            print("║" + line.ljust(78) + "║")

        print("╚" + "═" * 78 + "╝")

    @staticmethod
    def save_report(results: List[BacktestResult], filepath: str = "alpha_report.json"):
        """Save results to JSON."""
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_strategies": len(results),
            "passed_filter": sum(1 for r in results if r.passes_filter),
            "results": [r.to_dict() for r in sorted(results, key=lambda r: r.sharpe_ratio, reverse=True)],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\n📊 Report saved to: {filepath}")

    @staticmethod
    def get_top_k(results: List[BacktestResult], k: int = 5) -> List[BacktestResult]:
        """Return top-k strategies by Sharpe ratio that pass filters."""
        passing = [r for r in results if r.passes_filter]
        return sorted(passing, key=lambda r: r.sharpe_ratio, reverse=True)[:k]


# ═══════════════════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

class AlphaPipeline:
    """End-to-end pipeline: generate → backtest → rank → submit."""

    def __init__(self, competition_type: str = "futures",
                 api_key: str = "", output_dir: str = "quantiacs_workspace"):
        self.competition_type = competition_type
        self.api_key = api_key
        self.base_dir = Path(output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.generator = AlphaGenerator(
            competition_type=competition_type,
            output_dir=str(self.base_dir / "generated_alphas"),
            api_key=api_key,
        )
        self.backtester = BacktestEngine(timeout=600, api_key=api_key)
        self.submitter = SubmissionManager(
            api_key=api_key,
            output_dir=str(self.base_dir / "submissions"),
        )
        self.dashboard = ResultsDashboard()

    def run(self, num_alphas: int = 20, top_k: int = 5,
            templates: Optional[List[str]] = None,
            skip_backtest: bool = False) -> List[BacktestResult]:
        """Run the full pipeline."""
        print("\n" + "🚀 " * 20)
        print("  QUANTIACS ALPHA FACTORY PIPELINE")
        print("🚀 " * 20)
        print(f"\n  Competition: {self.competition_type}")
        print(f"  Target alphas: {num_alphas}")
        print(f"  Top-K to submit: {top_k}")

        # Step 1: Generate
        print(f"\n{'='*60}")
        print("  STEP 1: GENERATING ALPHA STRATEGIES")
        print(f"{'='*60}")
        strategies = self.generator.generate_batch(
            num_alphas=num_alphas, templates=templates
        )
        print(f"  ✅ Generated {len(strategies)} strategies")

        if skip_backtest:
            print("\n  ⏭️  Skipping backtest (--skip-backtest)")
            results = [
                BacktestResult(
                    strategy_name=Path(fp).stem,
                    template_name=tn,
                    params=p,
                    competition_type=self.competition_type,
                    file_path=fp,
                )
                for fp, tn, p in strategies
            ]
            return results

        # Step 2: Backtest
        print(f"\n{'='*60}")
        print("  STEP 2: BACKTESTING STRATEGIES")
        print(f"{'='*60}")
        results = self.backtester.run_batch(strategies)

        # Step 3: Rank & Report
        print(f"\n{'='*60}")
        print("  STEP 3: RANKING & REPORTING")
        print(f"{'='*60}")
        self.dashboard.print_summary(results)
        self.dashboard.save_report(
            results, str(self.base_dir / "alpha_report.json")
        )

        # Step 4: Prepare top-K for submission
        top_results = self.dashboard.get_top_k(results, top_k)
        if top_results:
            print(f"\n{'='*60}")
            print(f"  STEP 4: PREPARING TOP-{top_k} FOR SUBMISSION")
            print(f"{'='*60}")
            for r in top_results:
                self.submitter.prepare_submission(r.file_path, r.strategy_name)
        else:
            print(f"\n  ⚠️  No strategies passed the Sharpe ≥ 0.7 filter.")
            print(f"  Consider adjusting parameters or trying different templates.")

        return results

    def list_templates(self):
        """Print available templates for the competition type."""
        compatible = self.generator.get_compatible_templates()
        print(f"\n📋 Available templates for '{self.competition_type}':")
        print(f"{'─'*60}")
        for name in compatible:
            tpl = ALPHA_TEMPLATES[name]
            n_combos = 1
            for vals in tpl["param_grid"].values():
                n_combos *= len(vals)
            print(f"  [{tpl['category']:>15}] {name:<30} ({n_combos} combos)")
            print(f"                    {tpl['description']}")
        total_combos = sum(
            len(list(itertools.product(*ALPHA_TEMPLATES[n]["param_grid"].values())))
            for n in compatible
        )
        print(f"\n  Total possible combinations: {total_combos}")


# ═══════════════════════════════════════════════════════════════════
#  CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Quantiacs Alpha Factory — Auto-generate, backtest & submit trading alphas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # List available templates
          python quantiacs_alpha_factory.py --mode list --competition futures

          # Generate 20 alphas (no backtest)
          python quantiacs_alpha_factory.py --mode generate --competition futures --num-alphas 20

          # Full pipeline: generate → backtest → rank → prepare submission
          python quantiacs_alpha_factory.py --mode pipeline --competition futures --num-alphas 30 --top-k 5

          # Backtest a single strategy
          python quantiacs_alpha_factory.py --mode backtest --strategy path/to/strategy.py

          # Prepare submission for a strategy
          python quantiacs_alpha_factory.py --mode submit --strategy path/to/strategy.py --api-key YOUR_KEY
        """),
    )
    parser.add_argument("--mode", required=True, 
                        choices=["list", "generate", "backtest", "pipeline", "submit"],
                        help="Operation mode")
    parser.add_argument("--competition", default="futures",
                        choices=["futures", "stocks", "cryptodaily", "cryptofutures"],
                        help="Competition type (default: futures)")
    parser.add_argument("--num-alphas", type=int, default=20,
                        help="Number of alphas to generate (default: 20)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-K strategies to prepare for submission (default: 5)")
    parser.add_argument("--strategy", type=str, default="",
                        help="Path to a specific strategy file")
    parser.add_argument("--api-key", type=str, default="",
                        help="Quantiacs API key for submission")
    parser.add_argument("--output-dir", type=str, default="quantiacs_workspace",
                        help="Output directory (default: quantiacs_workspace)")
    parser.add_argument("--templates", nargs="*", default=None,
                        help="Specific template names to use")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip backtest step (generate only)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Backtest timeout in seconds (default: 600)")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("API_KEY", "") or os.environ.get("QUANTIACS_API_KEY", "")

    pipeline = AlphaPipeline(
        competition_type=args.competition,
        api_key=api_key,
        output_dir=args.output_dir,
    )
    pipeline.backtester.timeout = args.timeout

    if args.mode == "list":
        pipeline.list_templates()

    elif args.mode == "generate":
        strategies = pipeline.generator.generate_batch(
            num_alphas=args.num_alphas, templates=args.templates
        )
        print(f"\n✅ Generated {len(strategies)} strategies in: "
              f"{pipeline.generator.output_dir}")
        for fp, tname, params in strategies:
            print(f"  📄 {Path(fp).name}")

    elif args.mode == "backtest":
        if not args.strategy:
            print("❌ --strategy path required for backtest mode")
            sys.exit(1)
        result = pipeline.backtester.run_backtest(args.strategy)
        pipeline.dashboard.print_summary([result])

    elif args.mode == "pipeline":
        pipeline.run(
            num_alphas=args.num_alphas,
            top_k=args.top_k,
            templates=args.templates,
            skip_backtest=args.skip_backtest,
        )

    elif args.mode == "submit":
        if not args.strategy:
            print("❌ --strategy path required for submit mode")
            sys.exit(1)
        if not api_key:
            print("⚠️  No API key provided. Submission will be prepared without it.")
            print("   Set via --api-key or QUANTIACS_API_KEY env variable.")
        pipeline.submitter.prepare_submission(args.strategy)


if __name__ == "__main__":
    main()