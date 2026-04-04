#!/bin/bash

# Name of a separate log file just for crashes/restarts
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_LOG="training_monitor.log"

cd "$ROOT_DIR"

echo "=== Starting Auto-Restart Watchdog (Tmux Mode) ===" | tee -a "$MONITOR_LOG"

while true; do
    echo "----------------------------------------------------" >> "$MONITOR_LOG"
    echo "🚀 Launching Python Script at $(date)" | tee -a "$MONITOR_LOG"
    
    # Run the python script
    # -u unbuffers output so you see it immediately in tmux
    PYTHON_BIN="${PYTHON_BIN:-python}"
    "$PYTHON_BIN" -u train_gnn_model.py
    
    # Capture exit code
    EXIT_CODE=$?
    
    # Exit code 0 means the script finished nicely (reached end of main)
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Training Finished Successfully at $(date)!" | tee -a "$MONITOR_LOG"
        break
    else
        echo "⚠️ CRASH DETECTED (Exit Code: $EXIT_CODE) at $(date)." | tee -a "$MONITOR_LOG"
        echo "🔄 Restarting in 10 seconds... (Press Ctrl+C to stop)" | tee -a "$MONITOR_LOG"
        sleep 10
    fi
done