#!/usr/bin/env python3
"""
Quick Start — Programmatic usage of Quantiacs Alpha Factory.

This script demonstrates how to use the AlphaPipeline class directly
instead of the CLI interface.
"""
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantiacs_alpha_factory import (
    AlphaGenerator,
    AlphaPipeline,
    BacktestEngine,
    ResultsDashboard,
    SubmissionManager,
    ALPHA_TEMPLATES,
)


def example_1_list_templates():
    """List all available alpha templates for futures."""
    print("=" * 60)
    print("Example 1: List Templates")
    print("=" * 60)
    
    pipeline = AlphaPipeline(competition_type="futures")
    pipeline.list_templates()


def example_2_generate_only():
    """Generate strategies without backtesting."""
    print("\n" + "=" * 60)
    print("Example 2: Generate Strategies")
    print("=" * 60)
    
    gen = AlphaGenerator(competition_type="futures", output_dir="workspace/alphas")
    
    # Generate from specific templates only
    strategies = gen.generate_batch(
        num_alphas=10,
        templates=["sma_crossover", "rsi_reversion", "channel_breakout"]
    )
    
    for filepath, tname, params in strategies:
        print(f"  📄 {tname}: {params}")


def example_3_full_pipeline():
    """Run the full pipeline end-to-end."""
    print("\n" + "=" * 60)
    print("Example 3: Full Pipeline")
    print("=" * 60)
    
    # Set your API key
    api_key = os.environ.get("QUANTIACS_API_KEY", "")
    
    pipeline = AlphaPipeline(
        competition_type="futures",
        api_key=api_key,
        output_dir="workspace",
    )
    
    # Run pipeline (set skip_backtest=True if qnt not installed)
    results = pipeline.run(
        num_alphas=15,
        top_k=3,
        skip_backtest=True,  # Set to False if qnt is installed
    )
    
    print(f"\nGenerated {len(results)} strategies!")


def example_4_crypto_contest():
    """Generate strategies for the Crypto Top-10 Long-Only contest."""
    print("\n" + "=" * 60)
    print("Example 4: Crypto Long-Only Contest")
    print("=" * 60)
    
    gen = AlphaGenerator(
        competition_type="cryptodaily",
        output_dir="workspace/crypto_alphas",
    )
    
    strategies = gen.generate_batch(
        num_alphas=10,
        templates=["momentum_long_only", "equal_weight_liquid"]
    )
    
    for filepath, tname, params in strategies:
        print(f"  📄 {tname}: {params}")


def example_5_prepare_submission():
    """Prepare a strategy for submission."""
    print("\n" + "=" * 60)
    print("Example 5: Prepare Submission")
    print("=" * 60)
    
    # First, generate a strategy
    gen = AlphaGenerator(competition_type="futures", output_dir="workspace/alphas")
    result = gen.generate_strategy("sma_crossover", {"fast_period": 20, "slow_period": 200})
    
    if result:
        filepath, code = result
        sub = SubmissionManager(api_key="", output_dir="workspace/submissions")
        sub.prepare_submission(filepath)


if __name__ == "__main__":
    example_1_list_templates()
    example_2_generate_only()
    example_3_full_pipeline()
    example_4_crypto_contest()
    example_5_prepare_submission()
