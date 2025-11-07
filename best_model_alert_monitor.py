"""
Alert monitor using the best performing model: Very Conservative RF (5m).
Win Rate: 90.3%, Gap: 11.6%, Reward:Risk: 3:1
"""

from data_fetcher import AAVEDataFetcher
from feature_engineering import FeatureEngineer
from backtest_with_risk import RiskManagedBacktester
from risk_manager import RiskManager
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import subprocess
import time
from datetime import datetime
import os


def send_mac_notification(title, message):
    """Send native macOS notification."""
    try:
        # Format message for macOS (use | separator)
        message_clean = message.replace('\n', ' | ')
        script = f'''
        display notification "{message_clean}" with title "{title}" sound name "Glass"
        '''
        subprocess.run(['osascript', '-e', script], check=True, capture_output=True)
        return True
    except:
        return False


class BestModelAlertMonitor:
    """Alert monitor using the best performing model configuration."""
    
    def __init__(self):
        """Initialize with best model specs."""
        # Best Model: Very Conservative RF (5m)
        self.model_config = {
            'name': 'Very Conservative RF',
            'model_type': 'random_forest',
            'n_estimators': 30,
            'max_depth': 4,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt'
        }
        
        # Model specs
        self.timeframe = '5m'
        self.win_rate = 90.3
        self.overfitting_gap = 11.6
        self.reward_risk_ratio = 3.0
        self.commission = 0.0085  # 0.85% Robinhood
        self.position_size = 3500
        self.stop_loss_pct = 0.0133  # 1.33%
        self.target_gain_pct = 0.04  # 4%
        
        # Initialize components
        self.fetcher = AAVEDataFetcher(source='cryptocompare')
        self.engineer = FeatureEngineer()
        self.risk_manager = RiskManager(
            reward_risk_ratio=self.reward_risk_ratio,
            target_gain_pct=self.target_gain_pct,
            position_size=self.position_size
        )
        
        # Model and scaler
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Create model
        self._create_model()
    
    def _create_model(self):
        """Create the best model."""
        self.model = RandomForestClassifier(
            n_estimators=self.model_config['n_estimators'],
            max_depth=self.model_config['max_depth'],
            min_samples_split=self.model_config['min_samples_split'],
            min_samples_leaf=self.model_config['min_samples_leaf'],
            max_features=self.model_config['max_features'],
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    
    def _create_combined_features(self, df_features):
        """Create combined indicator features (from enhanced model)."""
        df = df_features.copy()
        
        # RSI + Stochastic confirmation
        df['rsi_stoch_bullish'] = ((df['rsi'] < 35) & (df['stoch_k'] < 30)).astype(int)
        df['momentum_strength'] = (df['rsi'] + df['stoch_k']) / 2
        
        # MACD + Moving Average trend confirmation
        df['macd_ma_bullish'] = ((df['macd'] > df['macd_signal']) & 
                                 (df['close'] > df['sma_50'])).astype(int)
        df['trend_strength'] = (df['macd_diff'] / df['close']) * 100
        
        # Bollinger Bands + Volume confirmation
        df['bb_volume_buy'] = ((df['bb_position'] < 0.2) & 
                              (df['volume_ratio'] > 1.2)).astype(int)
        
        # ADX filter
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        
        # Confluence score
        buy_score = (
            (df['rsi'] < 40).astype(int) * 2 +
            (df['stoch_k'] < 30).astype(int) * 2 +
            (df['macd'] > df['macd_signal']).astype(int) * 2 +
            (df['close'] > df['sma_50']).astype(int) * 1 +
            (df['bb_position'] < 0.3).astype(int) * 2 +
            (df['volume_ratio'] > 1.1).astype(int) * 1 +
            (df['adx'] > 25).astype(int) * 1
        )
        df['buy_confluence'] = buy_score
        
        return df
    
    def train_model(self, days=30):
        """Train the model on recent data."""
        print(f"Training best model ({self.model_config['name']}) on {self.timeframe} data...")
        
        # Fetch data
        df = self.fetcher.fetch_data(days=days, timeframe=self.timeframe)
        
        # If not enough real data, use synthetic from daily
        if df is None or len(df) < 100:
            print(f"  Not enough {self.timeframe} data, using synthetic from daily...")
            daily_fetcher = AAVEDataFetcher(source='coingecko')
            daily_df = daily_fetcher.fetch_data(days=365)
            if daily_df is not None:
                df = self.fetcher._resample_to_timeframe(daily_df, self.timeframe)
        
        if df is None or len(df) < 50:
            print(f"❌ Not enough data to train")
            return False
        
        # Feature engineering
        df_features = self.engineer.add_technical_indicators(df)
        df_combined = self._create_combined_features(df_features)
        
        # Create target
        threshold = 0.005  # 0.5% for 5m
        target = self.engineer.create_target(df_combined, prediction_horizon=1, threshold=threshold)
        df_combined['target'] = target
        features, target_clean = self.engineer.prepare_features(df_combined, target_col='target')
        
        # Clean features (remove infinity and NaN)
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Train
        X_scaled = self.scaler.fit_transform(features)
        self.model.fit(X_scaled, target_clean)
        self.is_trained = True
        
        print(f"✓ Model trained on {len(features)} samples")
        return True
    
    def check_for_signals(self):
        """Check for trading signals."""
        if not self.is_trained:
            if not self.train_model():
                return None
        
        # Fetch latest data - prioritize real data sources
        df = None
        
        # Try CryptoCompare first (real intraday data)
        try:
            df = self.fetcher.fetch_data(days=7, timeframe=self.timeframe)
        except:
            pass
        
        # If no real data or price seems wrong, try CoinGecko
        if df is None or len(df) < 20:
            try:
                daily_fetcher = AAVEDataFetcher(source='coingecko')
                daily_df = daily_fetcher.fetch_data(days=30)
                if daily_df is not None and len(daily_df) > 0:
                    # Verify price is reasonable (AAVE should be $100-$500 range)
                    current_daily_price = daily_df['close'].iloc[-1]
                    if 100 <= current_daily_price <= 500:
                        df = self.fetcher._resample_to_timeframe(daily_df, self.timeframe)
            except:
                pass
        
        if df is None or len(df) < 20:
            print("❌ Not enough data")
            return None
        
        # Verify price is reasonable before proceeding
        current_price = df['close'].iloc[-1]
        if current_price < 50 or current_price > 1000:
            print(f"⚠️  Price seems incorrect: ${current_price:.2f}")
            print("   Fetching fresh data from CoinGecko...")
            try:
                daily_fetcher = AAVEDataFetcher(source='coingecko')
                daily_df = daily_fetcher.fetch_data(days=1)
                if daily_df is not None and len(daily_df) > 0:
                    correct_price = daily_df['close'].iloc[-1]
                    if 100 <= correct_price <= 500:
                        # Use correct price
                        df['close'].iloc[-1] = correct_price
                        current_price = correct_price
                        print(f"   ✓ Using correct price: ${current_price:.2f}")
            except:
                pass
        
        # Feature engineering
        df_features = self.engineer.add_technical_indicators(df)
        df_combined = self._create_combined_features(df_features)
        
        # Get latest features
        features = self.engineer.prepare_features(df_combined)
        
        # Clean features (remove infinity and NaN)
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        latest_features = features.iloc[-1:].copy()
        latest_scaled = self.scaler.transform(latest_features)
        
        # Predict
        prediction = self.model.predict(latest_scaled)[0]
        probabilities = self.model.predict_proba(latest_scaled)[0]
        confidence = probabilities.max()
        
        # Get current price (use verified price)
        current_price = df['close'].iloc[-1]
        current_idx = features.index[-1]
        
        # Check if BUY signal
        if prediction == 1:  # BUY
            confluence = df_combined.loc[current_idx, 'buy_confluence'] if 'buy_confluence' in df_combined.columns else 0
            atr = df_combined.loc[current_idx, 'atr'] if 'atr' in df_combined.columns else None
            
            # Get trade plan
            plan = self.risk_manager.get_trade_plan(current_price, atr)
            
            # Recalculate with commission
            position_value, quantity = self.risk_manager.calculate_position_size(
                current_price, plan['stop_loss'], commission=self.commission
            )
            
            # Filter: only high-quality signals
            min_confidence = 0.50
            min_confluence = 5
            
            if confidence >= min_confidence and confluence >= min_confluence and plan['is_valid']:
                return {
                    'signal': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': plan['stop_loss'],
                    'take_profit': plan['take_profit'],
                    'position_value': position_value,
                    'quantity': quantity,
                    'confidence': confidence * 100,
                    'confluence': confluence,
                    'max_loss_pct': plan['max_loss_pct'] * 100,
                    'target_gain_pct': plan['target_gain_pct'] * 100,
                    'reward_risk_ratio': plan['reward_risk_ratio']
                }
        
        return None
    
    def send_alert(self, signal_data):
        """Send notification for trading signal."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = (
            f"Entry: ${signal_data['entry_price']:.2f} | "
            f"Stop: ${signal_data['stop_loss']:.2f} ({signal_data['max_loss_pct']:.2f}%) | "
            f"Target: ${signal_data['take_profit']:.2f} ({signal_data['target_gain_pct']:.2f}%) | "
            f"Size: ${signal_data['position_value']:,.0f} | "
            f"R:R {signal_data['reward_risk_ratio']:.1f}:1 | "
            f"Conf: {signal_data['confidence']:.0f}% | "
            f"Confl: {signal_data['confluence']}/11 | "
            f"Time: {current_time}"
        )
        
        title = "🎯 High-Quality BUY Signal (Best Model)"
        
        # Print
        print(f"\n{'='*70}")
        print(f"🔔 {title}")
        print(f"{message}")
        print(f"{'='*70}\n")
        
        # Send notification
        send_mac_notification(title, message)
    
    def run_check(self):
        """Run a single check."""
        signal = self.check_for_signals()
        if signal:
            self.send_alert(signal)
            return True
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Best Model Alert Monitor')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--interval', type=int, default=5, help='Check interval in minutes')
    
    args = parser.parse_args()
    
    monitor = BestModelAlertMonitor()
    
    print("="*70)
    print("BEST MODEL ALERT MONITOR")
    print("="*70)
    print(f"Model: {monitor.model_config['name']}")
    print(f"Timeframe: {monitor.timeframe}")
    print(f"Win Rate: {monitor.win_rate}%")
    print(f"Overfitting Gap: {monitor.overfitting_gap}%")
    print(f"Reward:Risk: {monitor.reward_risk_ratio}:1")
    print(f"Commission: {monitor.commission*100}% (Robinhood)")
    print(f"Position Size: ${monitor.position_size:,}")
    print("="*70)
    
    if args.once:
        monitor.run_check()
    else:
        print(f"\nMonitoring every {args.interval} minutes...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                monitor.run_check()
                time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            print("\n\nStopped by user")


if __name__ == "__main__":
    main()

