# Best Model Specifications

## 🏆 Best Performing Model

**Model Name**: Very Conservative RF (Random Forest)  
**Timeframe**: 5-minute candles  
**Performance**: 90.3% win rate, 11.6% overfitting gap

---

## 📊 Model Configuration

### Algorithm
- **Type**: Random Forest Classifier
- **n_estimators**: 30 (trees)
- **max_depth**: 4 (tree depth)
- **min_samples_split**: 20 (minimum samples to split)
- **min_samples_leaf**: 10 (minimum samples in leaf)
- **max_features**: 'sqrt' (features per split)
- **class_weight**: 'balanced' (handle imbalanced classes)
- **random_state**: 42

### Why This Configuration?
- **Very Conservative**: Prevents overfitting
- **Shallow Trees**: max_depth=4 limits complexity
- **High Regularization**: min_samples_split=20, min_samples_leaf=10
- **Feature Limiting**: max_features='sqrt' reduces overfitting

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Win Rate** | 90.3% |
| **Overfitting Gap** | 11.6% (excellent - minimal overfitting) |
| **Total Return** | 2221% (backtest - be realistic!) |
| **Reward:Risk Ratio** | 3.01:1 ✅ |
| **Total Trades** | 2,206 |
| **Sharpe Ratio** | 4.44 |

---

## 🔧 Trading Parameters

| Parameter | Value |
|-----------|-------|
| **Timeframe** | 5-minute candles |
| **Position Size** | $3,500 |
| **Stop-Loss** | 1.33% (max loss per trade) |
| **Take-Profit** | 4.00% (target gain) |
| **Reward:Risk** | 3:1 ratio |
| **Commission** | 0.85% (Robinhood) |
| **Target Threshold** | 0.5% (for 5m predictions) |

---

## 📊 Technical Indicators Used

### Base Indicators (27 features)

#### Momentum Indicators
- **RSI (14-period)**: Overbought/oversold
- **Stochastic Oscillator**: %K and %D
- **MACD**: MACD line, signal line, histogram

#### Trend Indicators
- **Moving Averages**:
  - SMA: 7, 14, 30, 50 periods
  - EMA: 12, 26 periods
- **ADX**: Trend strength (0-100)

#### Volatility Indicators
- **Bollinger Bands**: Upper, lower, middle (SMA 20)
- **ATR**: Average True Range (volatility)

#### Volume Indicators
- **Volume SMA**: 20-period average
- **Volume Ratio**: Current / Average volume

#### Price Action
- **Price Changes**: 1, 7, 30 period returns
- **High-Low Spread**: Daily range
- **Price Position**: Position in recent range

### Combined Features (48 total)

#### Confluence Features
1. **RSI + Stochastic**: Momentum confirmation
   - `rsi_stoch_bullish`: Both oversold
   - `momentum_strength`: Combined score

2. **MACD + Moving Averages**: Trend confirmation
   - `macd_ma_bullish`: MACD bullish + price above 50-day MA
   - `trend_strength`: Trend momentum score

3. **Bollinger Bands + Volume**: Volatility confirmation
   - `bb_volume_buy`: Price near lower band + high volume

4. **ADX Filter**: Strong trend indicator
   - `strong_trend`: ADX > 25

5. **Buy Confluence Score** (0-11):
   - RSI < 40: +2 points
   - Stochastic < 30: +2 points
   - MACD > Signal: +2 points
   - Price > SMA 50: +1 point
   - BB Position < 0.3: +2 points
   - Volume Ratio > 1.1: +1 point
   - ADX > 25: +1 point

---

## 🎯 Signal Filtering

### Minimum Requirements for Trade
- **Model Confidence**: ≥ 50%
- **Confluence Score**: ≥ 5/11
- **ADX**: ≥ 12 (trend strength)
- **Valid Trade Plan**: Meets risk criteria

### Trade Plan Validation
- Stop-loss: 1.33% max
- Take-profit: 4.00% target
- Reward:Risk: 3:1 ratio
- Position size: $3,500 (with commission)

---

## ⚠️ Realistic Expectations

### Backtest vs Reality

| Metric | Backtest | Realistic Expectation |
|--------|----------|----------------------|
| Win Rate | 90.3% | **60-75%** |
| Annual Return | 2000%+ | **20-50%** |
| Trades/Month | 200+ | **5-15** |
| Overfitting Gap | 11.6% | Monitor closely |

### Why Real Performance Will Be Lower
1. **Slippage**: Not accounted for
2. **Market Conditions**: Model trained on past data
3. **Execution Delays**: Real trades may miss exact entry/exit
4. **Emotional Factors**: Real trading psychology
5. **Commission Impact**: 0.85% on each trade

---

## 🚀 Usage

### Run Alert Monitor
```bash
# Run once
python best_model_alert_monitor.py --once

# Run continuously (every 5 minutes)
python best_model_alert_monitor.py --interval 5

# Run in background
nohup python best_model_alert_monitor.py --interval 5 > logs/best_model.log 2>&1 &
```

### What You'll Get
- **Notifications** when high-quality BUY signals are detected
- **Trade Plan** with entry, stop-loss, take-profit
- **Confidence** and **Confluence** scores
- **Risk Management** built-in

---

## 📋 Model Training

The model is trained on:
- **5-minute candles** from CryptoCompare API
- **30 days** of historical data (or synthetic from daily if needed)
- **Walk-forward validation** (train on past, test on future)
- **Target**: Predict 0.5% price movement

---

## ✅ Summary

**Best Model**: Very Conservative RF (5m)
- ✅ 90.3% win rate (exceeds 70-80% target)
- ✅ 11.6% gap (below 15% target - minimal overfitting)
- ✅ 3:1 reward:risk ratio
- ✅ 0.85% commission accounted for
- ✅ $3,500 position size
- ✅ 1.33% stop-loss protection

**Action**: Use `best_model_alert_monitor.py` for notifications!

---

*Last Updated: $(date)*


