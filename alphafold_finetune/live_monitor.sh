#!/bin/bash
# Live training monitor - follow logs in real-time

LOG_DIR="training_logs/enhanced_models_take6"
LOG_FILE="$LOG_DIR/training.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo ""
    echo "Available log directories:"
    ls -lh training_logs/
    echo ""
    echo "Usage: bash live_monitor.sh [log_directory]"
    echo "Example: bash live_monitor.sh enhanced_models_quick_test"
    exit 1
fi

# Override with custom directory if provided
if [ ! -z "$1" ]; then
    LOG_FILE="training_logs/$1/training.log"
    if [ ! -f "$LOG_FILE" ]; then
        LOG_FILE="training_logs/$1/quick_test.log"
    fi
fi

echo "==================================="
echo "   LIVE TRAINING MONITOR"
echo "==================================="
echo "Following: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo "==================================="
echo ""

# Highlight important patterns
tail -f "$LOG_FILE" | grep --line-buffered -E "(TRAIN Epoch|VALIDATION|collapse|SANITY CHECK|Prediction|WARNING|NEW BEST|std|range)" --color=always
