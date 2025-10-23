#!/bin/bash
# START HERE - Quick verification and training workflow

cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║          ALPHAFOLD FINETUNING - QUICK START GUIDE             ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

📋 WORKFLOW TO VERIFY IMPROVEMENTS QUICKLY:

┌────────────────────────────────────────────────────────────────┐
│  STEP 1: Quick Test (5-10 minutes)                            │
└────────────────────────────────────────────────────────────────┘
   Run a short test to verify the fixes work before long training:
   
   $ bash quick_test.sh
   
   ✅ If you see: "Sanity check PASSED" → proceed to Step 2
   ❌ If you see: "MODEL COLLAPSE DETECTED" → debug needed

┌────────────────────────────────────────────────────────────────┐
│  STEP 2: Full Training (with monitoring)                      │
└────────────────────────────────────────────────────────────────┘
   Start training in one terminal:
   
   $ bash train.sh
   
   In another terminal, monitor live:
   
   $ bash live_monitor.sh

┌────────────────────────────────────────────────────────────────┐
│  STEP 3: Automatic Sanity Check (step 50)                     │
└────────────────────────────────────────────────────────────────┘
   Training will automatically check at step 50:
   
   ✅ Prediction std > 0.5  → GOOD, continue training
   ❌ Prediction std < 0.5  → COLLAPSE, stop immediately

┌────────────────────────────────────────────────────────────────┐
│  STEP 4: Periodic Monitoring                                  │
└────────────────────────────────────────────────────────────────┘
   Check training progress periodically:
   
   $ bash monitor_training.sh

╔════════════════════════════════════════════════════════════════╗
║  KEY METRICS TO WATCH                                         ║
╚════════════════════════════════════════════════════════════════╝

✅ GOOD SIGNS:
   • Prediction std > 0.5 and increasing
   • Prediction range expanding (not stuck at 6-7)
   • MAE decreasing over time
   • Total Loss ≈ Regression Loss

❌ BAD SIGNS (STOP TRAINING):
   • Prediction std < 0.5 (constant predictions)
   • Predictions stuck in [6.0, 7.0] range
   • MAE not improving after 200+ steps
   • Total Loss >> Regression Loss

╔════════════════════════════════════════════════════════════════╗
║  WHAT'S BEEN FIXED                                            ║
╚════════════════════════════════════════════════════════════════╝

1. Loss Dominance: Regression loss now dominates when training
   only the binder (no structure loss interference)

2. Variance Penalty: Model penalized if predictions have low
   variance (prevents collapse to mean ~6.5)

3. Early Detection: Automatic sanity check at step 50 catches
   collapse immediately instead of after weeks

╔════════════════════════════════════════════════════════════════╗
║  TIME SAVINGS                                                 ║
╚════════════════════════════════════════════════════════════════╝

OLD WAY:  Train 1 week → realize broken → debug → repeat
          Total: 3-4 weeks wasted ❌

NEW WAY:  Test 10 min → fix → test 10 min → working!
          Then train 1 week successfully ✅

╔════════════════════════════════════════════════════════════════╗
║  AVAILABLE COMMANDS                                           ║
╚════════════════════════════════════════════════════════════════╝

bash quick_test.sh       → 10-min verification run
bash train.sh            → Full training
bash live_monitor.sh     → Real-time monitoring
bash monitor_training.sh → Periodic check summary

For detailed info, see: QUICK_VERIFICATION_GUIDE.md

EOF

echo ""
read -p "Press Enter to continue..."
