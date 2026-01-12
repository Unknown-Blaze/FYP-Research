#!/bin/bash

# ==============================================================================
# GPU AUTO-SCHEDULER & PIPELINE RUNNER
# 1. Waits for GPU > 8GB
# 2. Runs Inference (extract_predictions.py)
# 3. Runs Meta-Training (train_ensemble.py)
# ==============================================================================

# 1. CONFIGURATION
# ----------------
REQUIRED_FREE_MIB=14000
CHECK_INTERVAL=10

# Define your two steps here
CMD_STEP_1="python extract_predictions.py"
CMD_STEP_2="python train_ensemble.py"

echo "🔍 Monitoring GPUs for > ${REQUIRED_FREE_MIB} MiB of free memory..."
echo "📋 Pipeline: $CMD_STEP_1 -> $CMD_STEP_2"
echo "-------------------------------------------------------------"

while true; do
    # 2. GET GPU STATUS
    MEMORY_FREE_LIST=($(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits))

    # 3. CHECK EACH GPU
    FOUND_GPU=-1
    
    for i in "${!MEMORY_FREE_LIST[@]}"; do
        FREE_MEM=${MEMORY_FREE_LIST[$i]}
        if [ "$FREE_MEM" -ge "$REQUIRED_FREE_MIB" ]; then
            echo "✅ FOUND SPACE! GPU $i has ${FREE_MEM} MiB free."
            FOUND_GPU=$i
            break
        fi
    done

    # 4. LAUNCH OR WAIT
    if [ "$FOUND_GPU" -ne -1 ]; then
        echo "🚀 Starting Pipeline on GPU $FOUND_GPU..."
        echo "-------------------------------------------------------------"
        
        # Set the specific GPU for this script only
        export CUDA_VISIBLE_DEVICES=$FOUND_GPU
        
        # --- STEP 1: INFERENCE ---
        echo "▶️ [Step 1/2] Running: $CMD_STEP_1"
        $CMD_STEP_1
        
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "❌ CRITICAL ERROR: Step 1 failed (Exit Code: $EXIT_CODE). Stopping."
            exit $EXIT_CODE
        else
            echo "✅ Step 1 Finished Successfully."
        fi
        
        echo "-------------------------------------------------------------"
        
        # --- STEP 2: ENSEMBLE TRAINING ---
        # (This usually runs on CPU, but runs here to keep flow connected)
        echo "▶️ [Step 2/2] Running: $CMD_STEP_2"
        $CMD_STEP_2
        
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "❌ ERROR: Step 2 failed (Exit Code: $EXIT_CODE)."
        else
            echo "✅✅ PIPELINE COMPLETE! 🏆"
        fi
        
        # Break the loop since we are done
        break
        
    else
        # Print waiting status
        STATUS_STR=""
        for i in "${!MEMORY_FREE_LIST[@]}"; do
            STATUS_STR+="GPU$i: ${MEMORY_FREE_LIST[$i]}MiB | "
        done
        echo -ne "\r⏳ Waiting... ($STATUS_STR)"
        
        sleep $CHECK_INTERVAL
    fi
done