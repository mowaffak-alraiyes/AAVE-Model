# Paper Trading Guide

## 🤖 Automatic Paper Trading System

The paper trading system automatically executes trades based on your best model's signals. It runs entirely locally - no external API keys needed!

---

## 🚀 Quick Start

### Run Once (Test)
```bash
python paper_trading.py --once --balance 10000
```

### Run Continuously (Auto Trading)
```bash
python paper_trading.py --balance 10000 --interval 5
```

### Run in Background
```bash
./run_paper_trading.sh
```

**Stop background process:**
```bash
kill $(cat paper_trading.pid)
```

---

## 📊 How It Works

### 1. **Signal Detection**
- Checks for BUY signals every 5 minutes (configurable)
- Uses your best model: **Very Conservative RF (5m)**
- Only executes high-quality signals:
  - Confidence ≥ 50%
  - Confluence ≥ 5/11
  - Valid risk management

### 2. **Automatic Execution**
- **BUY Orders**: Automatically placed when signal detected
- **Position Size**: $3,500 (with 0.85% commission)
- **Stop-Loss**: 1.33% max loss
- **Take-Profit**: 4.00% target gain
- **Reward:Risk**: 3:1 ratio

### 3. **Risk Management**
- Monitors active positions continuously
- Automatically executes stop-loss if hit
- Automatically executes take-profit if hit
- Tracks all trades and performance

### 4. **Performance Tracking**
- Real-time portfolio value
- Win rate, total return
- Average win/loss
- Reward:Risk ratio
- Trade history saved to JSON files

---

## 📁 Files Created

The system creates these files to track your paper trading:

- `paper_trading_positions.json` - Current open positions
- `paper_trading_history.json` - All executed trades
- `paper_trading_orders.json` - Order log
- `logs/paper_trading.log` - System logs

---

## 💰 Example Output

```
✅ ORDER EXECUTED:
   Symbol: AAVE
   Side: BUY
   Quantity: 11.234567
   Price: $311.50
   Cost: $3,500.00
   Stop-Loss: $307.35
   Take-Profit: $323.96
   Balance: $6,500.00

🔄 TRADE CLOSED:
   Symbol: AAVE
   Side: SELL
   Quantity: 11.234567
   Price: $323.96
   P&L: $139.84 (4.00%)
   Balance: $10,139.84

📊 PERFORMANCE (14:30:00):
   Portfolio Value: $10,139.84
   Total Return: 1.40%
   Win Rate: 100.0%
   Trades: 1
   Avg Win: $139.84
   Avg Loss: $0.00
   R:R Ratio: 3.00:1
```

---

## ⚙️ Configuration

### Change Initial Balance
```bash
python paper_trading.py --balance 20000 --interval 5
```

### Change Check Interval
```bash
python paper_trading.py --balance 10000 --interval 10  # Check every 10 minutes
```

### Run Once (No Loop)
```bash
python paper_trading.py --once --balance 10000
```

---

## 📊 View Performance

### Check Current Status
The system prints performance every cycle. You can also check the JSON files:

```bash
# View positions
cat paper_trading_positions.json

# View trade history
cat paper_trading_history.json
```

### View Logs
```bash
tail -f logs/paper_trading.log
```

---

## 🎯 What Gets Traded

### Trade Criteria
- ✅ Model predicts BUY signal
- ✅ Confidence ≥ 50%
- ✅ Confluence score ≥ 5/11
- ✅ Valid risk management (stop-loss, take-profit)
- ✅ No existing position (won't double up)

### Trade Parameters
- **Entry**: Current market price
- **Stop-Loss**: 1.33% below entry
- **Take-Profit**: 4.00% above entry
- **Position Size**: $3,500
- **Commission**: 0.85% (Robinhood rate)

---

## ⚠️ Important Notes

1. **Local Only**: This runs entirely on your computer - no external APIs
2. **Paper Trading**: No real money at risk
3. **Market Data**: Uses CryptoCompare API (free, no key needed)
4. **Realistic**: Includes 0.85% commission on all trades
5. **Stop-Loss Protection**: Automatically limits losses to 1.33%

---

## 🔄 How It Monitors Trades

The system continuously checks:
1. **New Signals**: Every 5 minutes (or your interval)
2. **Active Positions**: Checks stop-loss/take-profit on every cycle
3. **Price Updates**: Fetches latest price to check triggers

If stop-loss or take-profit is hit, the trade is automatically closed.

---

## 📈 Performance Metrics

The system tracks:
- **Total Return**: Percentage gain/loss
- **Win Rate**: Percentage of winning trades
- **Average Win**: Average profit per winning trade
- **Average Loss**: Average loss per losing trade
- **Reward:Risk Ratio**: Average win / average loss
- **Portfolio Value**: Current total value (cash + positions)

---

## 🚨 Troubleshooting

### No Trades Executed
- Model may not be finding high-quality signals
- Check logs: `tail -f logs/paper_trading.log`
- Signals require confidence ≥ 50% and confluence ≥ 5/11

### Error Messages
- Check internet connection (needs CryptoCompare API)
- Check logs for details
- Make sure model trains successfully

### Reset Paper Trading
Delete the JSON files to start fresh:
```bash
rm paper_trading_*.json
```

---

## 🎯 Next Steps

1. **Start Paper Trading**: `./run_paper_trading.sh`
2. **Monitor Performance**: Watch the console output
3. **Review Trades**: Check `paper_trading_history.json`
4. **Analyze Results**: Compare to backtest performance
5. **Go Live**: Once confident, consider real trading (with caution!)

---

## ✅ Summary

**What You Get:**
- ✅ Automatic trade execution
- ✅ Risk management (stop-loss, take-profit)
- ✅ Performance tracking
- ✅ No API keys needed
- ✅ Runs locally on your computer

**How to Start:**
```bash
./run_paper_trading.sh
```

**That's it!** The system will automatically:
- Check for signals every 5 minutes
- Execute trades when high-quality signals are found
- Monitor positions and close at stop-loss/take-profit
- Track all performance metrics

---

*Happy Paper Trading! 📈*


