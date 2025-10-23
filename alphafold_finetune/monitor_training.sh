#!/bin/bash
# Quick training monitor - shows key metrics in real-time

LOG_FILE="training_logs/enhanced_models_take6/training.log"

echo "==================================="
echo "   TRAINING MONITOR"
echo "==================================="
echo ""

# Watch for collapse warning
echo "🔍 Checking for model collapse warnings..."
if grep -q "MODEL COLLAPSE DETECTED" "$LOG_FILE" 2>/dev/null; then
    echo "⚠️⚠️⚠️  MODEL COLLAPSE DETECTED!"
    grep -A 5 "MODEL COLLAPSE DETECTED" "$LOG_FILE" | tail -6
else
    echo "✅ No collapse warnings yet"
fi

echo ""
echo "-----------------------------------"

# Show early sanity check results
echo "📊 Early Sanity Check (Step 50):"
if grep -q "EARLY SANITY CHECK" "$LOG_FILE" 2>/dev/null; then
    grep -A 8 "EARLY SANITY CHECK" "$LOG_FILE" | tail -9
else
    echo "⏳ Not yet reached step 50"
fi

echo ""
echo "-----------------------------------"

# Show latest validation results
echo "📈 Latest Validation Results:"
grep "ENHANCED VALIDATION SUMMARY" -A 6 "$LOG_FILE" 2>/dev/null | tail -7

echo ""
echo "-----------------------------------"

# Show recent training metrics
echo "🎯 Recent Training Steps:"
tail -20 "$LOG_FILE" | grep "TRAIN Epoch"

echo ""
echo "==================================="
echo "Last updated: $(date)"
echo "==================================="
