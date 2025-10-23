#!/bin/bash
# Real-time prediction diversity monitor

LOG_FILE="training_logs/enhanced_models_quick_test/quick_test.log"

echo "==================================="
echo "  PREDICTION DIVERSITY MONITOR"
echo "==================================="
echo ""

while true; do
    if [ -f "$LOG_FILE" ]; then
        # Extract latest predictions
        LATEST=$(grep -oP "pred_pIC50=\K[0-9]+\.[0-9]+" "$LOG_FILE" | tail -50)
        
        if [ ! -z "$LATEST" ]; then
            # Calculate stats
            STATS=$(echo "$LATEST" | python3 -c "
import sys
import numpy as np
vals = [float(x.strip()) for x in sys.stdin if x.strip()]
if vals:
    print(f'Min: {min(vals):.2f} | Max: {max(vals):.2f} | Range: {max(vals)-min(vals):.2f} | Std: {np.std(vals):.2f} | Count: {len(vals)}')
else:
    print('No data yet...')
" 2>/dev/null)
            
            clear
            echo "==================================="
            echo "  PREDICTION DIVERSITY MONITOR"
            echo "  $(date '+%H:%M:%S')"
            echo "==================================="
            echo ""
            echo "📊 Last 50 predictions:"
            echo "   $STATS"
            echo ""
            echo "🎯 Target metrics:"
            echo "   Range: > 4.0 (currently: $(echo "$STATS" | grep -oP 'Range: \K[0-9.]+'))"
            echo "   Std:   > 1.5 (currently: $(echo "$STATS" | grep -oP 'Std: \K[0-9.]+'))"
            echo ""
            
            # Show last 5 predictions
            echo "📈 Recent predictions:"
            tail -5 "$LOG_FILE" | grep -E "TRAIN|VALIDATION" | grep -oP "(pred_pIC50|true_pIC50|mae)=[0-9]+\.[0-9]+" | tail -15
            
        fi
    else
        echo "⏳ Waiting for log file..."
    fi
    
    sleep 3
done
