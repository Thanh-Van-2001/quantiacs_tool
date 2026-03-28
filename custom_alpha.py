#!/usr/bin/env python3
"""
Custom Alpha Example — Add your own alpha signals to the factory.

This shows how to extend the Alpha Factory with proprietary alphas.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantiacs_alpha_factory import (
    register_alpha, _data_loader, _liquidity_filter,
    AlphaGenerator, AlphaPipeline, ALPHA_TEMPLATES,
)


# ══════════════════════════════════════════════════════════════
#  CUSTOM ALPHA 1: Dual Timeframe Momentum
# ══════════════════════════════════════════════════════════════

@register_alpha(
    name="dual_timeframe_momentum",
    category="custom",
    description="Combine short-term and long-term momentum signals",
    competition_types=["futures", "stocks"],
    param_grid={
        "short_mom": [5, 10, 20],
        "long_mom": [60, 120, 252],
        "blend_weight": [0.3, 0.5, 0.7],
    }
)
def gen_dual_timeframe(params, competition_type):
    sm = params["short_mom"]
    lm = params["long_mom"]
    bw = params["blend_weight"]
    if sm >= lm:
        return None
    return f'''
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr
import numpy as np

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    # Short-term momentum
    short_ret = (close.isel(time=-1) / close.isel(time=-{sm}) - 1)
    # Long-term momentum
    long_ret = (close.isel(time=-1) / close.isel(time=-{lm}) - 1)
    # Blend signals
    blended = {bw} * short_ret + (1 - {bw}) * long_ret
    weights = xr.where(blended > 0, 1, -1)
    {_liquidity_filter(competition_type)}
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={lm + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


# ══════════════════════════════════════════════════════════════
#  CUSTOM ALPHA 2: Volume-Weighted Trend
# ══════════════════════════════════════════════════════════════

@register_alpha(
    name="volume_weighted_trend",
    category="custom",
    description="Volume-weighted price trend detection",
    competition_types=["futures"],
    param_grid={
        "vwap_period": [10, 20, 40],
        "trend_period": [50, 100],
    }
)
def gen_volume_weighted_trend(params, competition_type):
    vp = params["vwap_period"]
    tp = params["trend_period"]
    return f'''
import qnt.ta as qnta
import qnt.data as qndata
import qnt.backtester as qnbk
import xarray as xr

def load_data(period):
    return qndata.{_data_loader(competition_type)}(tail=period)

def strategy(data):
    close = data.sel(field="close")
    vol = data.sel(field="vol")
    # Volume-weighted moving average
    vol_safe = vol.where(vol > 0).fillna(1)
    vwma = (close * vol_safe).rolling(time={vp}).sum() / vol_safe.rolling(time={vp}).sum()
    vwma_now = vwma.isel(time=-1)
    # Long-term trend
    ema_trend = qnta.ema(close, {tp}).isel(time=-1)
    cur = close.isel(time=-1)
    # Signal: price above VWMA and VWMA above trend
    bull = (cur > vwma_now) & (vwma_now > ema_trend)
    bear = (cur < vwma_now) & (vwma_now < ema_trend)
    weights = xr.where(bull, 1, xr.where(bear, -1, 0))
    return weights

qnbk.backtest(
    competition_type="{competition_type}",
    load_data=load_data,
    lookback_period={tp + 50},
    test_period=2 * 365,
    strategy=strategy,
    check_correlation=False,
)
'''


# ══════════════════════════════════════════════════════════════
#  USAGE: Generate strategies including custom alphas
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Custom alphas are now registered. Use them in the generator:
    gen = AlphaGenerator(competition_type="futures", output_dir="workspace/custom_alphas")
    
    # Show all templates including custom ones
    print("All available templates (including custom):")
    for name in gen.get_compatible_templates():
        tpl = ALPHA_TEMPLATES[name]
        print(f"  [{tpl['category']:>15}] {name}")
    
    # Generate only custom alphas
    strategies = gen.generate_batch(
        num_alphas=10,
        templates=["dual_timeframe_momentum", "volume_weighted_trend"],
    )
    
    print(f"\nGenerated {len(strategies)} custom strategies:")
    for fp, tname, params in strategies:
        print(f"  📄 {tname}: {params}")
