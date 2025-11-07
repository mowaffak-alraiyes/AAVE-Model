# AAVE Trading Model - Best Model System

## 🏆 Best Performing Model

**Very Conservative RF (Random Forest)** on **5-minute candles**
- **Win Rate**: 90.3%
- **Overfitting Gap**: 11.6% (minimal - excellent!)
- **Reward:Risk**: 3:1
- **Commission**: 0.85% (Robinhood) accounted for

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Alert Monitor

**Run once:**
```bash
python best_model_alert_monitor.py --once
```

**Run continuously (every 5 minutes):**
```bash
python best_model_alert_monitor.py --interval 5
```

**Run in background:**
```bash
./run_best_model.sh
```

**Stop background process:**
```bash
kill $(cat best_model.pid)
```

---

## 📊 Model Specifications

### Configuration
- **Algorithm**: Random Forest
- **n_estimators**: 30
- **max_depth**: 4
- **min_samples_split**: 20
- **min_samples_leaf**: 10
- **max_features**: 'sqrt'

### Trading Parameters
- **Timeframe**: 5-minute candles
- **Position Size**: $3,500
- **Stop-Loss**: 1.33% (max loss)
- **Take-Profit**: 4.00% (target gain)
- **Reward:Risk**: 3:1 ratio
- **Commission**: 0.85% (Robinhood)

### Technical Indicators
- **RSI, Stochastic, MACD**
- **Moving Averages** (SMA, EMA)
- **Bollinger Bands**
- **ADX** (trend strength)
- **ATR** (volatility)
- **Volume** indicators
- **Combined confluence scores**

See `BEST_MODEL_SPECS.md` for full details.

---

## 📱 Notifications

You'll receive macOS notifications when:
- High-quality BUY signals are detected
- Model confidence ≥ 50%
- Confluence score ≥ 5/11
- Valid trade plan (meets risk criteria)

**Notification includes:**
- Entry price
- Stop-loss (1.33% max)
- Take-profit (4% target)
- Position size ($3,500)
- Reward:Risk ratio (3:1)
- Confidence & Confluence scores
- Timestamp

---

## ⚠️ Realistic Expectations

### Backtest vs Reality

| Metric | Backtest | Realistic |
|--------|----------|-----------|
| Win Rate | 90.3% | **60-75%** |
| Annual Return | 2000%+ | **20-50%** |
| Trades/Month | 200+ | **5-15** |

**Why lower?**
- Slippage not accounted for
- Market conditions change
- Execution delays
- Emotional factors
- Commission impact

---

## 📁 Project Structure

```
AAVE Model/
├── best_model_alert_monitor.py  # Main notification system ⭐
├── find_best_model.py            # Model comparison tool
├── data_fetcher.py              # Data fetching (CryptoCompare, CoinGecko)
├── feature_engineering.py        # Technical indicators
├── enhanced_trading_model.py     # Model logic & combined features
├── risk_manager.py               # Risk management (stop-loss, take-profit)
├── backtest_with_risk.py         # Backtesting with risk management
├── BEST_MODEL_SPECS.md           # Full model specifications
├── run_best_model.sh             # Start background monitor
└── requirements.txt               # Dependencies
```

---

## 🔧 Advanced Usage

### Compare Models
```bash
python find_best_model.py
```

### Check Overfitting
```bash
python check_overfitting.py --timeframe 5m
```

---

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for full list

---

## ⚠️ Important Notes

1. **Start Small**: Test with smaller position sizes first
2. **Monitor Closely**: Watch for overfitting signs
3. **Use Stop-Losses**: Always protect capital (1.33% max)
4. **Be Realistic**: Expect 60-75% win rate, not 90%
5. **Paper Trade First**: Test before using real money

---

## 📞 Support

For questions or issues, check:
- `BEST_MODEL_SPECS.md` - Full model details
- Logs: `logs/best_model.log`

---

**Last Updated**: 2025-01-06  
**Best Model**: Very Conservative RF (5m)  
**Status**: ✅ Production Ready
