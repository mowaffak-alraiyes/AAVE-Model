"""
Backtest with Risk Management - Tests strategy with stop-loss and take-profit
"""

import pandas as pd
import numpy as np
from risk_manager import RiskManager
from enhanced_trading_model import EnhancedTradingModel
from data_fetcher import AAVEDataFetcher
from feature_engineering import FeatureEngineer


class RiskManagedBacktester:
    """Backtests with stop-loss, take-profit, and risk management."""
    
    def __init__(self, initial_capital=10000, reward_risk_ratio=3.0, 
                 target_gain_pct=0.04, position_size=3500, commission=0.0085):
        """
        Initialize risk-managed backtester.
        
        Args:
            initial_capital: Starting capital
            reward_risk_ratio: Target reward:risk ratio
            target_gain_pct: Target gain percentage
            position_size: Average position size
            commission: Trading commission (default 0.85% for Robinhood)
        """
        self.initial_capital = initial_capital
        self.risk_manager = RiskManager(
            reward_risk_ratio=reward_risk_ratio,
            target_gain_pct=target_gain_pct,
            position_size=position_size
        )
        self.commission = commission  # 0.85% for Robinhood
    
    def backtest(self, df, trade_plans):
        """
        Backtest with risk management.
        
        Args:
            df: DataFrame with price data
            trade_plans: List of trade plans from model
            
        Returns:
            Dictionary with results
        """
        capital = self.initial_capital
        active_trades = []  # List of open positions with stop-loss/take-profit
        closed_trades = []
        equity_curve = []
        
        for i in range(len(df)):
            current_date = df.index[i]
            current_price = df.iloc[i]['close']
            current_atr = df.iloc[i].get('atr', 0)
            
            # Check active trades for stop-loss or take-profit
            trades_to_close = []
            for trade in active_trades:
                # Check stop-loss
                if current_price <= trade['stop_loss']:
                    # Stop-loss hit (account for commission on exit)
                    exit_price = trade['stop_loss'] * (1 - self.commission)
                    pnl = (exit_price - trade['entry_price']) * trade['quantity']
                    pnl_pct = ((exit_price / trade['entry_price']) - 1) * 100
                    capital += exit_price * trade['quantity']
                    
                    closed_trades.append({
                        'entry_date': trade['entry_date'],
                        'exit_date': current_date,
                        'entry_price': trade['entry_price'],
                        'exit_price': trade['stop_loss'],
                        'quantity': trade['quantity'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'STOP_LOSS'
                    })
                    trades_to_close.append(trade)
                
                # Check take-profit
                elif current_price >= trade['take_profit']:
                    # Take-profit hit (account for commission on exit)
                    exit_price = trade['take_profit'] * (1 - self.commission)
                    pnl = (exit_price - trade['entry_price']) * trade['quantity']
                    pnl_pct = ((exit_price / trade['entry_price']) - 1) * 100
                    capital += exit_price * trade['quantity']
                    
                    closed_trades.append({
                        'entry_date': trade['entry_date'],
                        'exit_date': current_date,
                        'entry_price': trade['entry_price'],
                        'exit_price': trade['take_profit'],
                        'quantity': trade['quantity'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'TAKE_PROFIT'
                    })
                    trades_to_close.append(trade)
            
            # Remove closed trades
            for trade in trades_to_close:
                active_trades.remove(trade)
            
            # Check for new trade signals
            for plan in trade_plans:
                # Match dates (handle timestamp vs date comparison)
                plan_date = pd.Timestamp(plan['date']).normalize() if hasattr(plan['date'], 'normalize') else pd.Timestamp(plan['date'])
                current_date_normalized = pd.Timestamp(current_date).normalize() if hasattr(current_date, 'normalize') else pd.Timestamp(current_date)
                
                if plan_date == current_date_normalized and plan.get('is_valid', True):
                    # Check if we have enough capital (including commission)
                    # Position value already includes commission in risk_manager
                    required_capital = plan['position_value']
                    
                    if capital >= required_capital:
                        # Enter trade (commission already included in position_value)
                        capital -= required_capital
                        active_trades.append({
                            'entry_date': current_date,
                            'entry_price': plan['entry_price'] * (1 + self.commission),  # Effective entry with commission
                            'stop_loss': plan['stop_loss'],
                            'take_profit': plan['take_profit'],
                            'quantity': plan['quantity'],
                            'risk_amount': plan['risk_amount']
                        })
            
            # Calculate current equity
            equity = capital
            for trade in active_trades:
                equity += current_price * trade['quantity']
            
            equity_curve.append({
                'date': current_date,
                'equity': equity,
                'capital': capital,
                'open_positions': len(active_trades),
                'price': current_price
            })
        
        # Close any remaining positions at final price (account for commission)
        final_price = df.iloc[-1]['close'] * (1 - self.commission)  # Commission on exit
        for trade in active_trades:
            pnl = (final_price - trade['entry_price']) * trade['quantity']
            pnl_pct = ((final_price / trade['entry_price']) - 1) * 100
            capital += final_price * trade['quantity']
            
            closed_trades.append({
                'entry_date': trade['entry_date'],
                'exit_date': df.index[-1],
                'entry_price': trade['entry_price'],
                'exit_price': final_price,
                'quantity': trade['quantity'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': 'END_OF_PERIOD'
            })
        
        # Calculate final equity
        final_equity = capital
        
        # Analyze results
        trades_df = pd.DataFrame(closed_trades)
        
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) * 100
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            total_return = ((final_equity / self.initial_capital) - 1) * 100
            
            # Calculate actual reward:risk ratio
            if avg_loss != 0:
                actual_ratio = abs(avg_win / avg_loss)
            else:
                actual_ratio = 0
            
            equity_df = pd.DataFrame(equity_curve)
            equity_df.set_index('date', inplace=True)
            
            returns = equity_df['equity'].pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            
            peak = equity_df['equity'].expanding().max()
            drawdown = (equity_df['equity'] - peak) / peak
            max_drawdown = drawdown.min() * 100
            
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            total_return = 0
            actual_ratio = 0
            sharpe_ratio = 0
            max_drawdown = 0
            equity_df = pd.DataFrame(equity_curve)
            if len(equity_df) > 0:
                equity_df.set_index('date', inplace=True)
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades) if len(trades_df) > 0 else 0,
            'losing_trades': len(losing_trades) if len(trades_df) > 0 else 0,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'reward_risk_ratio': actual_ratio,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades_df,
            'equity_curve': equity_df
        }


def test_enhanced_strategy(days=365):
    """Test the enhanced strategy with risk management."""
    print("="*70)
    print("ENHANCED STRATEGY WITH RISK MANAGEMENT")
    print("="*70)
    print("Target: 3-5% gain per trade, 3:1 reward:risk, stop-loss protection")
    print(f"Position size: $3,500 average\n")
    
    # Fetch data
    print("[1/4] Fetching data...")
    fetcher = AAVEDataFetcher()
    df = fetcher.fetch_data(days=days)
    
    if df is None:
        print("Failed to fetch data")
        return
    
    # Feature engineering
    print("[2/4] Engineering features...")
    engineer = FeatureEngineer()
    df_features = engineer.add_technical_indicators(df)
    
    # Enhanced model with combined features
    print("[3/4] Training enhanced model...")
    model = EnhancedTradingModel()
    df_combined = model.create_combined_features(df_features)
    
    # Prepare features
    target = engineer.create_target(df_combined, prediction_horizon=1, threshold=0.02)
    df_combined['target'] = target
    features, target_clean = engineer.prepare_features(df_combined, target_col='target')
    
    # Train
    split_idx = int(len(features) * 0.8)
    X_train = features.iloc[:split_idx]
    y_train = target_clean.iloc[:split_idx]
    
    model.train(X_train, y_train, test_size=0.0, use_time_split=False)
    
    # Get predictions with risk management
    print("[4/4] Generating trade plans with risk management...")
    test_features = features.iloc[split_idx:]
    test_prices = df_combined.iloc[split_idx:]
    
    results = model.predict_with_risk_management(test_features, test_prices)
    
    print(f"\nGenerated {len(results['trade_plans'])} valid trade plans")
    print(f"Filtered from {len(results['predictions'][results['predictions'] == 1])} raw BUY signals")
    
    # Backtest
    print("\nRunning backtest with risk management...")
    backtester = RiskManagedBacktester(
        initial_capital=10000,
        reward_risk_ratio=3.0,
        target_gain_pct=0.04,
        position_size=3500
    )
    
    backtest_results = backtester.backtest(test_prices, results['trade_plans'])
    
    # Print results
    print("\n" + "="*70)
    print("RISK-MANAGED BACKTEST RESULTS")
    print("="*70)
    print(f"Initial Capital: ${backtest_results['initial_capital']:,.2f}")
    print(f"Final Equity: ${backtest_results['final_equity']:,.2f}")
    print(f"Total Return: {backtest_results['total_return']:.2f}%")
    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {backtest_results['total_trades']}")
    print(f"  Winning Trades: {backtest_results['winning_trades']}")
    print(f"  Losing Trades: {backtest_results['losing_trades']}")
    print(f"  Win Rate: {backtest_results['win_rate']:.1f}%")
    print(f"\nRisk Metrics:")
    print(f"  Average Win: ${backtest_results['avg_win']:,.2f}")
    print(f"  Average Loss: ${backtest_results['avg_loss']:,.2f}")
    print(f"  Reward:Risk Ratio: {backtest_results['reward_risk_ratio']:.2f}:1")
    print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {backtest_results['max_drawdown']:.2f}%")
    
    if len(backtest_results['trades']) > 0:
        print(f"\nTrade Breakdown:")
        exit_reasons = backtest_results['trades']['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count}")
    
    return backtest_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test enhanced strategy with risk management')
    parser.add_argument('--days', type=int, default=365, help='Days of historical data')
    
    args = parser.parse_args()
    test_enhanced_strategy(days=args.days)

