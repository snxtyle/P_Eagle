#!/bin/bash
# P-EAGLE Training Watchdog - Waits for rsync, then auto-starts training with OOM protection

LOG_FILE="$HOME/p_eagle_watchdog.log"
CHECKPOINT_DIR="/home/suraj/Desktop/P_Eagle/outputs/training_run"
PROJECT_DIR="/home/suraj/Desktop/P_Eagle"
VENV_PYTHON="/home/suraj/Desktop/P_Eagle/venv/bin/python"
TRAIN_SCRIPT="$VENV_PYTHON -m p_eagle.training.trainer"
PID_FILE="/tmp/p_eagle_training_pid"
RSYNC_PID_FILE="/tmp/p_eagle_rsync_pid"
FEATURE_DIR="/home/suraj/Desktop/P_Eagle/data/features/combined"

# Training parameters - For 32768 seq len
BATCH_SIZE=1
GRAD_ACCUM=64
SHARD_CACHE=2
NUM_EPOCHS=3

TRAIN_PARAMS="
    --drafter_model google/gemma-3-270m-it
    --target_hidden_dim 2560
    --feature_dir $FEATURE_DIR
    --output_dir $CHECKPOINT_DIR
    --use_lora
    --lora_rank 64
    --lora_alpha 256
    --learning_rate 5e-5
    --num_epochs $NUM_EPOCHS
    --speculation_depth 4
    --max_seq_len 32768
    --shard_cache_size $SHARD_CACHE
    --use_flash_attention
    --save_every 100
    --gpu-safety-margin 2.0
    --yes
"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

count_shards() {
    ls "$FEATURE_DIR"/*_shard*.pt 2>/dev/null | wc -l
}

wait_for_rsync() {
    log "Checking for rsync process..."

    # Check if rsync is currently running
    local rsync_pid=$(pgrep -f "rsync.*part1.*combined" | head -1)

    if [ -n "$rsync_pid" ]; then
        log "Found rsync running (PID: $rsync_pid) - waiting for it to finish..."
        echo "$rsync_pid" > "$RSYNC_PID_FILE"

        # Wait for rsync to finish
        while ps -p "$rsync_pid" > /dev/null 2>&1; do
            local current_shards=$(count_shards)
            log "  rsync still running... ($current_shards shards copied so far)"
            sleep 60  # Check every minute
        done

        local final_shards=$(count_shards)
        log "rsync finished! Total shards: $final_shards"
        rm -f "$RSYNC_PID_FILE"
    else
        log "No rsync process found"

        # Check how many shards are already available
        local current_shards=$(count_shards)
        log "Current shards in $FEATURE_DIR: $current_shards"

        # Expected: ~63 from part0 + ~58 from part1 = ~121 total
        if [ "$current_shards" -lt 60 ]; then
            log "WARNING: Less than 60 shards found. You may need to start rsync manually:"
            log "  rsync -avzP --ignore-existing suraj@10.8.0.105:/home/suraj/Desktop/P_Eagle/data/features/part1/ $FEATURE_DIR/"
        fi
    fi
}

find_latest_checkpoint() {
    # Find the latest checkpoint
    local latest=$(ls -td "$CHECKPOINT_DIR"/checkpoint_step_* 2>/dev/null | head -1)
    if [ -n "$latest" ] && [ -d "$latest" ]; then
        echo "$latest"
    else
        echo ""
    fi
}

start_training() {
    local resume_arg=""
    local checkpoint=$(find_latest_checkpoint)

    cd "$PROJECT_DIR"

    # Build dynamic train params with current settings
    local current_params="$TRAIN_PARAMS --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --shard_cache_size $SHARD_CACHE"

    if [ -n "$checkpoint" ]; then
        resume_arg="--resume $checkpoint"
        log "Found checkpoint: $checkpoint"
    else
        log "No checkpoint found, starting fresh"
    fi

    log "========================================="
    log "Starting training with:"
    log "  batch_size=$BATCH_SIZE, grad_accum=$GRAD_ACCUM"
    log "  shard_cache=$SHARD_CACHE, epochs=$NUM_EPOCHS"
    log "  shards available: $(count_shards)"
    log "========================================="

    $TRAIN_SCRIPT $current_params $resume_arg >> "$LOG_FILE" 2>&1 &
    TRAINING_PID=$!
    echo $TRAINING_PID > "$PID_FILE"
    log "Training started with PID: $TRAINING_PID"
}

reduce_batch_size() {
    if [ "$BATCH_SIZE" -gt 1 ]; then
        BATCH_SIZE=$((BATCH_SIZE - 1))
        GRAD_ACCUM=$((GRAD_ACCUM * 2))
        log "Reduced batch size to $BATCH_SIZE, grad_accum to $GRAD_ACCUM"
    elif [ "$SHARD_CACHE" -gt 1 ]; then
        SHARD_CACHE=$((SHARD_CACHE - 1))
        log "Reduced shard_cache_size to $SHARD_CACHE (batch_size already at 1)"
    elif [ "$GRAD_ACCUM" -lt 256 ]; then
        GRAD_ACCUM=$((GRAD_ACCUM * 2))
        log "Increased grad_accum to $GRAD_ACCUM (shard_cache already at 1)"
    else
        log "Cannot reduce further - already at minimum settings"
        return 1
    fi
    return 0
}

check_oom_in_logs() {
    tail -200 "$LOG_FILE" 2>/dev/null | grep -qiE "OOM|out of memory|Killed|SIGKILL|exit code 137"
}

monitor_training() {
    local crash_count=0
    local max_restarts=10
    local iteration=0

    while true; do
        iteration=$((iteration + 1))

        # Check if training process is still running
        if [ -f "$PID_FILE" ]; then
            current_pid=$(cat "$PID_FILE")
            if ! ps -p "$current_pid" > /dev/null 2>&1; then
                log "Training process died (was PID: $current_pid)"

                # Check for OOM/Killed in logs
                if check_oom_in_logs; then
                    log "OOM/Memory issue detected - will reduce batch size"
                    reduce_batch_size
                fi

                # Check exit code
                if wait "$current_pid" 2>/dev/null; then
                    log "Training completed normally!"
                    break
                fi

                crash_count=$((crash_count + 1))

                if [ $crash_count -ge $max_restarts ]; then
                    log "Too many crashes ($crash_count). Stopping watchdog."
                    log "Please investigate the issue manually."
                    log "Check logs at: $LOG_FILE"
                    break
                fi

                log "Restarting training in 10 seconds... (crash #$crash_count)"
                sleep 10
                start_training
            else
                if [ $((iteration % 12)) -eq 0 ]; then
                    local shards=$(count_shards)
                    log "Training running (PID: $current_pid) [batch=$BATCH_SIZE, accum=$GRAD_ACCUM, shards=$SHARD_CACHE, available=$shards]"
                fi
            fi
        else
            # No PID file, check if training is running
            running_pid=$(pgrep -f "trainer" | head -1)
            if [ -n "$running_pid" ]; then
                echo "$running_pid" > "$PID_FILE"
                log "Found running training (PID: $running_pid)"
            else
                log "No training running, starting..."
                start_training
            fi
        fi

        sleep 30  # Check every 30 seconds
    done
}

# Main
log "========================================="
log "P-EAGLE Training Watchdog Started"
log "========================================="

# Create PID file dir if needed
mkdir -p "$(dirname "$PID_FILE")"

# Step 1: Wait for rsync to finish
wait_for_rsync

# Step 2: Check if training is already running
running_pid=$(pgrep -f "p_eagle.training.trainer" | head -1)
if [ -n "$running_pid" ]; then
    log "Training already running - monitoring (PID: $running_pid)"
    echo "$running_pid" > "$PID_FILE"
    monitor_training
else
    log "Starting training now..."
    start_training
    monitor_training
fi