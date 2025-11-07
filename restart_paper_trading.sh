#!/bin/bash
# Restart paper trading with notifications enabled

echo "🔄 Restarting Paper Trading System..."
echo ""

# Stop current process if running
if [ -f paper_trading.pid ]; then
    PID=$(cat paper_trading.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "🛑 Stopping current process (PID: $PID)..."
        kill $PID
        sleep 2
    fi
    rm -f paper_trading.pid
fi

# Start fresh
echo "🚀 Starting with notifications enabled..."
./run_paper_trading.sh

echo ""
echo "✅ Restarted! You'll now get notifications for:"
echo "   📈 Trade executions"
echo "   🔄 Trade closures"
echo "   📊 Performance updates"


