#!/bin/bash
# Run the best model alert monitor in the background

echo "🚀 Starting Best Model Alert Monitor..."
echo "Model: Very Conservative RF (5m)"
echo "Win Rate: 90.3%, Gap: 11.6%"
echo ""

# Create logs directory
mkdir -p logs

# Run in background
nohup python3 best_model_alert_monitor.py --interval 5 > logs/best_model.log 2>&1 &

# Save PID
echo $! > best_model.pid

echo "✅ Alert monitor started (PID: $(cat best_model.pid))"
echo "📋 Logs: logs/best_model.log"
echo "🛑 Stop with: kill \$(cat best_model.pid)"


