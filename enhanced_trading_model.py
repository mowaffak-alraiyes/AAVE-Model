"""
Enhanced Trading Model with Indicator Combinations and Risk Management
Focuses on minimizing losses and maintaining 3:1 reward:risk ratio
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
from risk_manager import RiskManager
from feature_engineering import FeatureEngineer
import warnings
warnings.filterwarnings('ignore')


class EnhancedTradingModel:
    """Enhanced model that combines indicators intelligently with risk management."""
    
    def __init__(self, model_type='random_forest'):
        """Initialize enhanced trading model."""
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.risk_manager = RiskManager(
            reward_risk_ratio=3.0,
            target_gain_pct=0.04,  # 4% target (3-5% range)
            position_size=3500  # Average 3-4k
        )
        
        if model_type == 'random_forest':
            # Reduced complexity to prevent overfitting
            self.model = RandomForestClassifier(
                n_estimators=100,  # Reduced from 150
                max_depth=8,       # Reduced from 12 (less complex trees)
                min_samples_split=10,  # Increased from 5 (more regularization)
                min_samples_leaf=5,    # Increased from 2 (more regularization)
                max_features='sqrt',   # Limit features per split
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Handle imbalanced classes
            )
        elif model_type == 'gradient_boosting':
            # Reduced complexity to prevent overfitting
            self.model = GradientBoostingClassifier(
                n_estimators=100,  # Reduced from 150
                max_depth=4,       # Reduced from 6
                learning_rate=0.05,  # Reduced from 0.08 (more conservative)
                subsample=0.8,     # Add subsampling for regularization
                random_state=42
            )
    
    def create_combined_features(self, df_features):
        """
        Create combined indicator features that work together.
        
        Args:
            df_features: DataFrame with all indicators
            
        Returns:
            DataFrame with additional combined features
        """
        df = df_features.copy()
        
        # 1. RSI + Stochastic confirmation (momentum combo)
        df['rsi_stoch_bullish'] = ((df['rsi'] < 35) & (df['stoch_k'] < 30)).astype(int)
        df['rsi_stoch_bearish'] = ((df['rsi'] > 65) & (df['stoch_k'] > 70)).astype(int)
        df['momentum_strength'] = (df['rsi'] + df['stoch_k']) / 2
        
        # 2. MACD + Moving Average trend confirmation
        df['macd_ma_bullish'] = ((df['macd'] > df['macd_signal']) & 
                                 (df['close'] > df['sma_50'])).astype(int)
        df['macd_ma_bearish'] = ((df['macd'] < df['macd_signal']) & 
                                (df['close'] < df['sma_50'])).astype(int)
        df['trend_strength'] = (df['macd_diff'] / df['close']) * 100
        
        # 3. Bollinger Bands + Volume confirmation
        df['bb_volume_buy'] = ((df['bb_position'] < 0.2) & 
                              (df['volume_ratio'] > 1.2)).astype(int)
        df['bb_volume_sell'] = ((df['bb_position'] > 0.8) & 
                               (df['volume_ratio'] > 1.2)).astype(int)
        
        # 4. ADX filter (only trade in strong trends)
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        df['weak_trend'] = (df['adx'] < 20).astype(int)
        
        # 5. ATR-based volatility filter
        df['high_volatility'] = (df['atr'] > df['atr'].rolling(20).mean() * 1.5).astype(int)
        df['low_volatility'] = (df['atr'] < df['atr'].rolling(20).mean() * 0.7).astype(int)
        
        # 6. Multi-indicator confluence score
        # Higher score = stronger buy signal
        buy_score = (
            (df['rsi'] < 40).astype(int) * 2 +
            (df['stoch_k'] < 30).astype(int) * 2 +
            (df['macd'] > df['macd_signal']).astype(int) * 2 +
            (df['close'] > df['sma_50']).astype(int) * 1 +
            (df['bb_position'] < 0.3).astype(int) * 2 +
            (df['volume_ratio'] > 1.1).astype(int) * 1 +
            (df['adx'] > 25).astype(int) * 1
        )
        
        sell_score = (
            (df['rsi'] > 60).astype(int) * 2 +
            (df['stoch_k'] > 70).astype(int) * 2 +
            (df['macd'] < df['macd_signal']).astype(int) * 2 +
            (df['close'] < df['sma_50']).astype(int) * 1 +
            (df['bb_position'] > 0.7).astype(int) * 2 +
            (df['volume_ratio'] > 1.1).astype(int) * 1 +
            (df['adx'] > 25).astype(int) * 1
        )
        
        df['buy_confluence'] = buy_score
        df['sell_confluence'] = sell_score
        df['signal_strength'] = buy_score - sell_score
        
        return df
    
    def filter_signals(self, predictions, probabilities, features_df, min_confluence=6, min_confidence=0.65):
        """
        Filter signals to only high-quality trades.
        
        Args:
            predictions: Raw predictions
            probabilities: Prediction probabilities (numpy array or Series)
            features_df: Features dataframe with confluence scores
            min_confluence: Minimum confluence score required
            min_confidence: Minimum model confidence required
            
        Returns:
            Filtered predictions
        """
        filtered = predictions.copy()
        
        # Handle probabilities - convert to max confidence per sample
        if isinstance(probabilities, pd.Series):
            confidence = probabilities
        else:
            # Numpy array - get max probability for each sample
            import numpy as np
            confidence = pd.Series(np.max(probabilities, axis=1), index=predictions.index)
        
        # Only keep signals with:
        # 1. High model confidence
        # 2. Strong indicator confluence
        # 3. Reasonable trend (ADX > 15, more lenient)
        
        for i in range(len(predictions)):
            if predictions.iloc[i] == 1:  # BUY signal
                confluence = features_df.iloc[i].get('buy_confluence', 0)
                adx = features_df.iloc[i].get('adx', 0)
                conf = confidence.iloc[i]
                
                # More lenient filtering - use OR logic instead of AND
                # Keep signal if: (confluence OK) OR (confidence high) OR (both decent)
                confluence_ok = confluence >= min_confluence
                confidence_ok = conf >= min_confidence
                adx_ok = adx >= 12  # Lower ADX requirement
                
                # Keep if at least 2 out of 3 criteria are met
                criteria_met = sum([confluence_ok, confidence_ok, adx_ok])
                if criteria_met < 2:
                    filtered.iloc[i] = 0  # Change to HOLD
            
            elif predictions.iloc[i] == -1:  # SELL signal
                confluence = features_df.iloc[i].get('sell_confluence', 0)
                adx = features_df.iloc[i].get('adx', 0)
                conf = confidence.iloc[i]
                
                # More lenient filtering - use OR logic instead of AND
                confluence_ok = confluence >= min_confluence
                confidence_ok = conf >= min_confidence
                adx_ok = adx >= 12  # Lower ADX requirement
                
                # Keep if at least 2 out of 3 criteria are met
                criteria_met = sum([confluence_ok, confidence_ok, adx_ok])
                if criteria_met < 2:
                    filtered.iloc[i] = 0  # Change to HOLD
        
        return filtered
    
    def train(self, X, y, test_size=0.2, use_time_split=True):
        """Train the enhanced model."""
        if test_size == 0.0:
            # No test split, use all data for training
            X_train, y_train = X, y
            X_test, y_test = None, None
        elif use_time_split:
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        
        print(f"Training Accuracy: {train_acc:.4f}")
        
        if X_test is not None and y_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            test_pred = self.model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, test_pred)
            print(f"Test Accuracy: {test_acc:.4f}")
            return {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            }
        else:
            return {
                'train_accuracy': train_acc,
                'test_accuracy': None
            }
    
    def predict_with_risk_management(self, features_df, price_df):
        """
        Make predictions with risk management.
        
        Args:
            features_df: Features dataframe
            price_df: Price dataframe with ATR
            
        Returns:
            Dictionary with predictions and trade plans
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get raw predictions
        features_scaled = self.scaler.transform(features_df)
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Filter signals
        predictions_series = pd.Series(predictions, index=features_df.index)
        probabilities_series = pd.Series(probabilities.max(axis=1), index=features_df.index)
        
        # Adjust thresholds based on timeframe (shorter timeframes need more lenient filtering)
        # For shorter timeframes, we want more signals
        min_confluence = 2  # Very lenient for short timeframes
        min_confidence = 0.45  # Lower confidence for more signals
        
        filtered_predictions = self.filter_signals(
            predictions_series, 
            probabilities_series,
            features_df,
            min_confluence=min_confluence,
            min_confidence=min_confidence
        )
        
        # Create trade plans for BUY signals
        trade_plans = []
        for i, (idx, pred) in enumerate(filtered_predictions.items()):
            if pred == 1:  # BUY signal
                try:
                    entry_price = price_df.loc[idx, 'close']
                    atr = price_df.loc[idx, 'atr'] if 'atr' in price_df.columns else None
                    
                    plan = self.risk_manager.get_trade_plan(entry_price, atr)
                    
                    # Recalculate position size with commission
                    commission = 0.0085  # 0.85% Robinhood fee
                    position_value, quantity = self.risk_manager.calculate_position_size(
                        entry_price, plan['stop_loss'], commission=commission
                    )
                    plan['position_value'] = position_value
                    plan['quantity'] = quantity
                    plan['risk_amount'] = (entry_price * (1 + commission) - plan['stop_loss']) * quantity
                    
                    plan['date'] = idx
                    plan['signal'] = 'BUY'
                    plan['confidence'] = probabilities_series.loc[idx]
                    
                    # Get confluence score (handle if column doesn't exist)
                    if 'buy_confluence' in features_df.columns:
                        plan['confluence'] = features_df.loc[idx, 'buy_confluence']
                    else:
                        plan['confluence'] = 0
                    
                    # Accept trade if valid OR if confidence is high enough
                    if plan['is_valid'] or probabilities_series.loc[idx] >= 0.65:
                        trade_plans.append(plan)
                except Exception as e:
                    # Skip if there's an error
                    continue
        
        return {
            'predictions': filtered_predictions,
            'probabilities': probabilities_series,
            'trade_plans': trade_plans
        }
    
    def save(self, filepath):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'risk_manager': self.risk_manager
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        if 'risk_manager' in model_data:
            self.risk_manager = model_data['risk_manager']
        self.is_trained = True
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test enhanced model
    from data_fetcher import AAVEDataFetcher
    from feature_engineering import FeatureEngineer
    
    print("Testing Enhanced Trading Model...")
    fetcher = AAVEDataFetcher()
    df = fetcher.fetch_data(days=365)
    
    if df is not None:
        engineer = FeatureEngineer()
        df_features = engineer.add_technical_indicators(df)
        
        model = EnhancedTradingModel()
        df_combined = model.create_combined_features(df_features)
        print(f"Created {len(df_combined.columns)} features (including combined)")

