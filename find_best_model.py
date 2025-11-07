"""
Find best model configuration for 30m and 5m timeframes.
Tests different model combinations to achieve 70-80% win rate without overfitting.
Accounts for 0.85% Robinhood commission.
"""

from data_fetcher import AAVEDataFetcher
from feature_engineering import FeatureEngineer
from backtest_with_risk import RiskManagedBacktester
from risk_manager import RiskManager
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime
import json


# Expanded model configurations to test
MODEL_CONFIGS = [
    # Very Conservative (prevent overfitting)
    {
        'name': 'Very Conservative RF',
        'model_type': 'random_forest',
        'n_estimators': 30,
        'max_depth': 4,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 'sqrt'
    },
    {
        'name': 'Conservative RF',
        'model_type': 'random_forest',
        'n_estimators': 50,
        'max_depth': 5,
        'min_samples_split': 15,
        'min_samples_leaf': 8,
        'max_features': 'sqrt'
    },
    {
        'name': 'Moderate RF',
        'model_type': 'random_forest',
        'n_estimators': 75,
        'max_depth': 6,
        'min_samples_split': 12,
        'min_samples_leaf': 6,
        'max_features': 'sqrt'
    },
    {
        'name': 'Balanced RF',
        'model_type': 'random_forest',
        'n_estimators': 100,
        'max_depth': 8,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt'
    },
    # Gradient Boosting variants
    {
        'name': 'Very Conservative GB',
        'model_type': 'gradient_boosting',
        'n_estimators': 30,
        'max_depth': 2,
        'learning_rate': 0.03,
        'subsample': 0.7
    },
    {
        'name': 'Conservative GB',
        'model_type': 'gradient_boosting',
        'n_estimators': 50,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.8
    },
    {
        'name': 'Moderate GB',
        'model_type': 'gradient_boosting',
        'n_estimators': 75,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8
    },
]


def create_model_from_config(config):
    """Create a model from configuration."""
    if config['model_type'] == 'random_forest':
        return RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            min_samples_leaf=config['min_samples_leaf'],
            max_features=config.get('max_features', 'sqrt'),
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    else:
        return GradientBoostingClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            subsample=config.get('subsample', 1.0),
            random_state=42
        )


def test_model_on_timeframe(timeframe='30m', config=None, use_synthetic_if_needed=True):
    """
    Test a model configuration on a specific timeframe.
    Uses real data if available, synthetic if needed.
    """
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']} on {timeframe}")
    print(f"{'='*70}")
    
    # Try to get real data
    if timeframe in ['30m', '5m']:
        fetcher = AAVEDataFetcher(source='cryptocompare')
        max_days = 7 if timeframe == '5m' else 40
        df = fetcher.fetch_data(days=max_days, timeframe=timeframe)
        
        # If not enough real data, use synthetic from daily
        if df is None or len(df) < 200:
            if use_synthetic_if_needed:
                print(f"  Not enough real data ({len(df) if df is not None else 0} candles)")
                print(f"  Using synthetic data from daily candles for testing...")
                daily_fetcher = AAVEDataFetcher(source='coingecko')
                daily_df = daily_fetcher.fetch_data(days=365)
                if daily_df is not None:
                    # Generate synthetic shorter timeframe data
                    df = fetcher._resample_to_timeframe(daily_df, timeframe)
                    print(f"  ✓ Generated {len(df)} synthetic {timeframe} candles")
            else:
                print(f"  ❌ Not enough data")
                return None
    else:
        fetcher = AAVEDataFetcher(source='coingecko')
        df = fetcher.fetch_data(days=365)
    
    if df is None or len(df) < 100:
        print(f"  ❌ Not enough data")
        return None
    
    print(f"  ✓ Using {len(df)} candles")
    
    # Feature engineering
    engineer = FeatureEngineer()
    df_features = engineer.add_technical_indicators(df)
    
    # Create combined features (simplified version)
    from enhanced_trading_model import EnhancedTradingModel
    temp_model = EnhancedTradingModel()
    df_combined = temp_model.create_combined_features(df_features)
    
    # Prepare for training
    threshold_map = {
        '5m': 0.005,
        '30m': 0.01,
        '1d': 0.02
    }
    threshold = threshold_map.get(timeframe, 0.01)
    
    target = engineer.create_target(df_combined, prediction_horizon=1, threshold=threshold)
    df_combined['target'] = target
    features, target_clean = engineer.prepare_features(df_combined, target_col='target')
    
    # Walk-forward validation
    split_ratio = 0.7
    split_idx = int(len(features) * split_ratio)
    X_train = features.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_train = target_clean.iloc[:split_idx]
    y_test = target_clean.iloc[split_idx:]
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Check if we have multiple classes
    unique_classes = y_train.unique()
    if len(unique_classes) < 2:
        print(f"  ⚠️  Only {len(unique_classes)} class(es) in training data, skipping")
        return None
    
    # Create and train model
    model = create_model_from_config(config)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    try:
        model.fit(X_train_scaled, y_train)
    except ValueError as e:
        if "1 class" in str(e):
            print(f"  ⚠️  Model training failed: {e}")
            return None
        raise
    
    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    gap = train_acc - test_acc
    
    print(f"  Train Acc: {train_acc:.2%}, Test Acc: {test_acc:.2%}, Gap: {gap:.2%}")
    
    # Get predictions with filtering
    try:
        test_probs = model.predict_proba(X_test_scaled)
        test_confidence = test_probs.max(axis=1)
    except:
        # If predict_proba fails, use simple confidence
        test_confidence = np.ones(len(test_pred)) * 0.5
    
    # Filter: only high confidence predictions (but be more lenient)
    min_confidence = 0.50  # Lower threshold to get more trades
    high_conf_mask = test_confidence >= min_confidence
    filtered_predictions = test_pred.copy()
    filtered_predictions[~high_conf_mask] = 0  # Change low-confidence to HOLD
    
    # Create trade plans for BUY signals
    test_prices = df_combined.iloc[split_idx:]
    risk_manager = RiskManager(
        reward_risk_ratio=3.0,
        target_gain_pct=0.04,
        position_size=3500
    )
    
    trade_plans = []
    for i, (idx, pred) in enumerate(zip(X_test.index, filtered_predictions)):
        if pred == 1:  # BUY signal
            try:
                entry_price = test_prices.loc[idx, 'close']
                atr = test_prices.loc[idx, 'atr'] if 'atr' in test_prices.columns else None
                
                plan = risk_manager.get_trade_plan(entry_price, atr)
                
                # Account for 0.85% commission
                commission = 0.0085
                position_value, quantity = risk_manager.calculate_position_size(
                    entry_price, plan['stop_loss'], commission=commission
                )
                plan['position_value'] = position_value
                plan['quantity'] = quantity
                plan['risk_amount'] = (entry_price * (1 + commission) - plan['stop_loss']) * quantity
                plan['date'] = idx
                try:
                    plan['confidence'] = test_confidence[i] if i < len(test_confidence) else 0.5
                except:
                    plan['confidence'] = 0.5
                
                # Accept if valid OR if confidence is high enough
                if plan['is_valid'] or plan['confidence'] >= 0.55:
                    trade_plans.append(plan)
            except:
                continue
    
    if len(trade_plans) == 0:
        print(f"  ⚠️  No valid trades")
        return {
            'config_name': config['name'],
            'timeframe': timeframe,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': gap,
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'reward_risk_ratio': 0
        }
    
    # Backtest with 0.85% commission
    backtester = RiskManagedBacktester(
        initial_capital=10000,
        reward_risk_ratio=3.0,
        target_gain_pct=0.04,
        position_size=3500,
        commission=0.0085  # 0.85% Robinhood
    )
    
    backtest_results = backtester.backtest(test_prices, trade_plans)
    
    print(f"  Trades: {backtest_results['total_trades']}, "
          f"Win Rate: {backtest_results['win_rate']:.1f}%, "
          f"Return: {backtest_results['total_return']:.2f}%, "
          f"Gap: {gap:.2%}")
    
    return {
        'config_name': config['name'],
        'timeframe': timeframe,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': gap,
        'total_trades': backtest_results['total_trades'],
        'win_rate': backtest_results['win_rate'],
        'total_return': backtest_results['total_return'],
        'sharpe_ratio': backtest_results['sharpe_ratio'],
        'max_drawdown': backtest_results['max_drawdown'],
        'reward_risk_ratio': backtest_results['reward_risk_ratio'],
        'avg_win': backtest_results['avg_win'],
        'avg_loss': backtest_results['avg_loss']
    }


def find_best_models():
    """Find best model configurations for 30m and 5m."""
    print("="*70)
    print("FINDING BEST MODELS FOR 30m & 5m TIMEFRAMES")
    print("="*70)
    print("Target: 70-80% win rate, gap < 15%, 3:1 reward:risk")
    print("Commission: 0.85% (Robinhood)")
    print("Position Size: $3,500")
    print("="*70)
    
    all_results = []
    timeframes = ['30m', '5m']
    
    for timeframe in timeframes:
        print(f"\n\n{'#'*70}")
        print(f"TESTING {timeframe.upper()} TIMEFRAME")
        print(f"{'#'*70}")
        
        for config in MODEL_CONFIGS:
            result = test_model_on_timeframe(timeframe=timeframe, config=config)
            if result:
                all_results.append(result)
    
    # Analysis
    if len(all_results) == 0:
        print("\n❌ No results to analyze")
        return
    
    results_df = pd.DataFrame(all_results)
    
    print("\n\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n📊 ALL MODELS:")
    print(f"{'Model':<25} {'TF':<6} {'Win%':<8} {'Return%':<10} {'Gap%':<8} {'Trades':<8}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        gap_pct = row['gap'] * 100 if not pd.isna(row['gap']) else 0
        print(f"{row['config_name']:<25} {row['timeframe']:<6} "
              f"{row['win_rate']:>6.1f}% {row['total_return']:>8.2f}% "
              f"{gap_pct:>6.1f}% {row['total_trades']:>6}")
    
    # Find best models (70-80% win rate, gap < 15%)
    print(f"\n✅ BEST MODELS (70-80% win rate, gap < 15%):")
    print("-" * 70)
    
    best = results_df[
        (results_df['win_rate'] >= 70) & 
        (results_df['win_rate'] <= 80) &
        (results_df['gap'] < 0.15)
    ].sort_values('win_rate', ascending=False)
    
    if len(best) > 0:
        print(f"{'Model':<25} {'TF':<6} {'Win%':<8} {'Return%':<10} {'Gap%':<8} {'R:R':<8}")
        print("-" * 70)
        for _, row in best.iterrows():
            gap_pct = row['gap'] * 100
            print(f"{row['config_name']:<25} {row['timeframe']:<6} "
                  f"{row['win_rate']:>6.1f}% {row['total_return']:>8.2f}% "
                  f"{gap_pct:>6.1f}% {row['reward_risk_ratio']:>5.2f}:1")
    else:
        print("⚠️  No models meet all criteria (70-80% win, gap < 15%)")
        
        # Show closest
        print("\n📈 CLOSEST MATCHES:")
        print("-" * 70)
        
        # Best win rate with reasonable gap
        best_win = results_df[
            (results_df['win_rate'] >= 60) &
            (results_df['gap'] < 0.25)
        ].sort_values('win_rate', ascending=False).head(5)
        
        for _, row in best_win.iterrows():
            gap_pct = row['gap'] * 100
            print(f"  {row['config_name']} ({row['timeframe']}): "
                  f"{row['win_rate']:.1f}% win, {gap_pct:.1f}% gap, "
                  f"{row['total_return']:.2f}% return")
    
    # Save results
    results_df.to_csv('best_model_results.csv', index=False)
    print(f"\n✅ Results saved to best_model_results.csv")
    
    return results_df


if __name__ == "__main__":
    find_best_models()

