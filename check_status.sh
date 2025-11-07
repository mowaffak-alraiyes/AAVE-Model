#!/bin/bash
# Quick status check for paper trading

echo "📊 Paper Trading Status Check"
echo "============================"
echo ""

# Check if PID file exists
if [ -f paper_trading.pid ]; then
    PID=$(cat paper_trading.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Status: RUNNING (PID: $PID)"
    else
        echo "❌ Status: STOPPED (process not found)"
    fi
else
    echo "❌ Status: NOT STARTED (no PID file)"
fi

echo ""

# Check for recent activity
if [ -f logs/paper_trading.log ]; then
    echo "📋 Recent Activity:"
    tail -10 logs/paper_trading.log | grep -v "NotOpenSSLWarning" | grep -v "^$" | tail -5
    echo ""
fi

# Check for trades
if [ -f paper_trading_positions.json ]; then
    echo "📈 Open Positions:"
    cat paper_trading_positions.json | python3 -m json.tool 2>/dev/null || echo "  (checking...)"
    echo ""
fi

if [ -f paper_trading_history.json ]; then
    TRADES=$(cat paper_trading_history.json | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data))" 2>/dev/null || echo "0")
    echo "💰 Total Trades: $TRADES"
    echo ""
fi

echo "💡 To view live logs: tail -f logs/paper_trading.log"
echo "💡 To stop: kill \$(cat paper_trading.pid)"

