"""
Feature engineering module for creating technical indicators
and features for the AAVE trading model.
"""

import pandas as pd
import numpy as np
import ta


class FeatureEngineer:
    """Creates technical indicators and features for trading models."""
    
    def __init__(self):
        """Initialize feature engineer."""
        pass
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Ensure we have the required columns
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")
        if 'volume' not in df.columns:
            df['volume'] = 0  # Default volume if not available
        
        # Moving Averages
        df['sma_7'] = ta.trend.SMAIndicator(df['close'], window=7).sma_indicator()
        df['sma_14'] = ta.trend.SMAIndicator(df['close'], window=14).sma_indicator()
        df['sma_30'] = ta.trend.SMAIndicator(df['close'], window=30).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # RSI (Relative Strength Index)
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        
        # ATR (Average True Range)
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Volume indicators
        if df['volume'].sum() > 0:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_ratio'] = df['volume_ratio'].fillna(1)
        else:
            df['volume_sma'] = 0
            df['volume_ratio'] = 1
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_7'] = df['close'].pct_change(7)
        df['price_change_30'] = df['close'].pct_change(30)
        
        # Volatility
        df['volatility'] = df['price_change'].rolling(window=14).std()
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Price position in range
        df['price_position'] = (df['close'] - df['low'].rolling(14).min()) / (
            df['high'].rolling(14).max() - df['low'].rolling(14).min()
        )
        
        return df
    
    def create_target(self, df, prediction_horizon=1, threshold=0.02):
        """
        Create target variable for prediction.
        
        Args:
            df: DataFrame with price data
            prediction_horizon: Number of periods ahead to predict
            threshold: Minimum price change to consider a signal (2% default)
            
        Returns:
            Series with target values: 1 for buy, -1 for sell, 0 for hold
        """
        # Calculate future returns
        future_return = df['close'].shift(-prediction_horizon) / df['close'] - 1
        
        # Create signals
        target = pd.Series(0, index=df.index)
        target[future_return > threshold] = 1  # Buy signal
        target[future_return < -threshold] = -1  # Sell signal
        
        return target
    
    def prepare_features(self, df, target_col=None):
        """
        Prepare final feature set for model training.
        
        Args:
            df: DataFrame with all indicators
            target_col: Name of target column (if creating target)
            
        Returns:
            Tuple of (features_df, target_series) or just features_df
        """
        # Select feature columns (exclude OHLCV and target)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'market_cap']
        if target_col:
            exclude_cols.append(target_col)
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        features_df = df[feature_cols].copy()
        
        # Drop rows with NaN values
        features_df = features_df.dropna()
        
        if target_col:
            target = df.loc[features_df.index, target_col]
            return features_df, target
        else:
            return features_df


if __name__ == "__main__":
    # Test feature engineering
    from data_fetcher import AAVEDataFetcher
    
    fetcher = AAVEDataFetcher()
    df = fetcher.fetch_data(days=365)
    
    if df is not None:
        engineer = FeatureEngineer()
        df_features = engineer.add_technical_indicators(df)
        target = engineer.create_target(df_features, prediction_horizon=1)
        
        features, target_clean = engineer.prepare_features(df_features, target_col='target')
        df_features['target'] = target
        
        print(f"Created {len(features.columns)} features")
        print(f"Feature columns: {list(features.columns)}")
        print(f"\nTarget distribution:")
        print(target_clean.value_counts())

