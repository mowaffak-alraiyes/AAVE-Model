#!/bin/bash
# Run automatic paper trading system

echo "🤖 Starting Auto Paper Trading System..."
echo ""

# Create logs directory
mkdir -p logs

# Check if dependencies are installed
if ! python3 -c "import pandas" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    python3 -m pip install -q pandas numpy scikit-learn requests ta
fi

# Run in background
nohup python3 paper_trading.py --balance 10000 --interval 5 > logs/paper_trading.log 2>&1 &

# Save PID
echo $! > paper_trading.pid

echo "✅ Paper trading started (PID: $(cat paper_trading.pid))"
echo "📋 Logs: logs/paper_trading.log"
echo "🛑 Stop with: kill \$(cat paper_trading.pid)"
echo ""
echo "💡 The system will:"
echo "   • Check for signals every 5 minutes"
echo "   • Automatically execute BUY orders"
echo "   • Monitor stop-loss and take-profit"
echo "   • Track performance"

