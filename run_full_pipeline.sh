#!/bin/bash
# P-EAGLE Full Pipeline with Watchdog
# - Starts/resumes training
# - Monitors for crashes and restarts
# - Detects epoch completion and auto-optimizes

set -e

# ============== CONFIGURATION ==============
PROJECT_DIR="/home/suraj/Desktop/P_Eagle"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
TRAIN_SCRIPT="$VENV_PYTHON -m p_eagle.training.trainer"

# Paths
CHECKPOINT_DIR="$PROJECT_DIR/outputs/training_5k"
LOG_FILE="$CHECKPOINT_DIR/pipeline_watchdog.log"
PID_FILE="/tmp/p_eagle_training_pid"
EPOCH_MARKER_FILE="$CHECKPOINT_DIR/.epoch_completed"

# Training configuration
DRAFTER_MODEL="google/gemma-3-4b-it"
TARGET_HIDDEN_DIM=2560
FEATURE_DIR="$PROJECT_DIR/data/features/extraction_5k"
OUTPUT_DIR="$CHECKPOINT_DIR"

# Watchdog settings
MAX_RESTARTS=10
CRASH_RESTART_DELAY=10
CHECK_INTERVAL=30
EPOCH_CHECK_INTERVAL=60

# ============== FUNCTIONS ==============
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"
    log "$1"
    echo "=========================================" | tee -a "$LOG_FILE"
}

# Get latest checkpoint number
get_latest_step() {
    local latest=$(ls -td "$CHECKPOINT_DIR"/checkpoint_step_* 2>/dev/null | head -1 | sed 's|.*checkpoint_step_||')
    echo "${latest:-0}"
}

# Get latest checkpoint path
get_latest_checkpoint() {
    ls -td "$CHECKPOINT_DIR"/checkpoint_step_* 2>/dev/null | head -1
}

# Check if training process is running
is_training_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        ps -p "$pid" > /dev/null 2>&1
        return $?
    fi
    return 1
}

# Start training
start_training() {
    log "Starting training..."

    # Determine resume point
    RESUME_FLAG=""
    if [ -d "$CHECKPOINT_DIR/epoch_1_checkpoint" ]; then
        # Resume from epoch 1 checkpoint for optimized training
        RESUME_FLAG="--resume $CHECKPOINT_DIR/epoch_1_checkpoint"
        log "Resuming from epoch_1_checkpoint"
    else
        local latest=$(get_latest_checkpoint)
        if [ -n "$latest" ] && [ -d "$latest" ]; then
            RESUME_FLAG="--resume $latest"
            log "Resuming from $latest"
        else
            log "Starting fresh training"
        fi
    fi

    # Build training command
    TRAIN_CMD="$TRAIN_SCRIPT \
        --drafter_model $DRAFTER_MODEL \
        --target_hidden_dim $TARGET_HIDDEN_DIM \
        --feature_dir $FEATURE_DIR \
        --output_dir $OUTPUT_DIR \
        --use_lora \
        --lora_rank 32 \
        --lora_alpha 128 \
        --learning_rate 5e-5 \
        --num_epochs 3 \
        --speculation_depth 4 \
        --max_seq_len 4096 \
        --use_flash_attention \
        --save_every 50 \
        --gpu-safety-margin 1.0 \
        --yes \
        --shard_cache_size 2 \
        $RESUME_FLAG"

    # Start in tmux
    tmux new-session -d -s p_eagle_training "cd $PROJECT_DIR && $TRAIN_CMD"

    # Get the actual Python PID
    sleep 2
    local python_pid=$(pgrep -f "p_eagle.training.trainer" | head -1)
    if [ -n "$python_pid" ]; then
        echo "$python_pid" > "$PID_FILE"
        log "Training started in tmux (PID: $python_pid)"
    else
        log "Warning: Could not find training PID"
    fi
}

# Stop training
stop_training() {
    log "Stopping training..."
    tmux kill-session -t p_eagle_training 2>/dev/null || true
    pkill -f "p_eagle.training.trainer" 2>/dev/null || true
    sleep 3
    pkill -9 -f "p_eagle.training.trainer" 2>/dev/null || true
    sleep 2
    log "Training stopped"
}

# Monitor for crashes and restart
monitor_crashes() {
    local crash_count=0
    local iteration=0

    while true; do
        iteration=$((iteration + 1))

        if ! is_training_running; then
            log "Training process died (crash #$((crash_count + 1)))"

            # Check if training completed successfully
            if [ -f "$CHECKPOINT_DIR/training_complete" ]; then
                log "Training completed successfully!"
                return 0
            fi

            crash_count=$((crash_count + 1))

            if [ $crash_count -ge $MAX_RESTARTS ]; then
                log "ERROR: Too many crashes ($crash_count). Stopping watchdog."
                return 1
            fi

            log "Restarting training in ${CRASH_RESTART_DELAY}s... (crash #$crash_count)"
            sleep "$CRASH_RESTART_DELAY"
            start_training
        else
            # Log progress every 12 iterations (~6 minutes)
            if [ $((iteration % 12)) -eq 0 ]; then
                local step=$(get_latest_step)
                local checkpoint_count=$(ls "$CHECKPOINT_DIR"/checkpoint_step_* 2>/dev/null | wc -l)
                log "Training running... [step=$step, checkpoints=$checkpoint_count, crash_count=$crash_count]"
            fi
        fi

        sleep "$CHECK_INTERVAL"
    done
}

# Watch for epoch 1 completion and auto-optimize
watch_epoch_completion() {
    local epoch1_end_step=6700
    local check_count=0

    log "Watching for epoch 1 completion (step > $epoch1_end_step)..."

    while true; do
        local current_step=$(get_latest_step)

        if [ "$current_step" -gt "$epoch1_end_step" ]; then
            log ""
            log "========================================="
            log "EPOCH 1 COMPLETED! (step $current_step > $epoch1_end_step)"
            log "========================================="

            # Wait for epoch checkpoint to be fully saved
            log "Waiting for epoch checkpoint to stabilize..."
            sleep 60

            # Check if epoch_1_checkpoint exists
            if [ -d "$CHECKPOINT_DIR/epoch_1_checkpoint" ]; then
                log "Epoch 1 checkpoint found at $CHECKPOINT_DIR/epoch_1_checkpoint"

                # Stop current training
                stop_training

                # Mark epoch 1 as completed
                echo "epoch1_completed=$(date)" > "$EPOCH_MARKER_FILE"

                log "========================================="
                log "Ready for optimized training!"
                log "To start optimized training:"
                log "  cd $PROJECT_DIR"
                log "  ./run_full_pipeline.sh"
                log "========================================="

                # Return to trigger main monitoring loop to restart with optimized params
                return 0
            else
                log "Warning: epoch_1_checkpoint not found yet, continuing to monitor..."
            fi
        else
            check_count=$((check_count + 1))
            # Log every 10 checks (~10 minutes)
            if [ $((check_count % 10)) -eq 0 ]; then
                local checkpoint_count=$(ls "$CHECKPOINT_DIR"/checkpoint_step_* 2>/dev/null | wc -l)
                log "Progress: step=$current_step/$epoch1_end_step ($checkpoint_count checkpoints)"
            fi
        fi

        sleep "$EPOCH_CHECK_INTERVAL"
    done
}

# ============== MAIN ==============
main() {
    mkdir -p "$(dirname "$PID_FILE")"
    mkdir -p "$CHECKPOINT_DIR"

    log_section "P-EAGLE Full Pipeline Started"
    log "Checkpoint dir: $CHECKPOINT_DIR"
    log "Log file: $LOG_FILE"
    log "PID file: $PID_FILE"

    # Check for existing training
    if is_training_running; then
        local pid=$(cat "$PID_FILE" 2>/dev/null)
        log "Training already running (PID: $pid)"
        log "Use 'tmux attach -t p_eagle_training' to view"
    else
        log "No training running, starting..."
        start_training
    fi

    # Main loop: monitor for crashes and epoch completion
    while true; do
        log_section "Starting watchdog monitor"

        # Check if epoch 1 is completed and we should optimize
        if [ -d "$CHECKPOINT_DIR/epoch_1_checkpoint" ] && [ ! -f "$EPOCH_MARKER_FILE" ]; then
            watch_epoch_completion
            # After epoch 1 completion, the watchdog will restart training
            # with the epoch_1_checkpoint as resume point
        fi

        # Monitor for crashes
        if ! monitor_crashes; then
            log "Watchdog stopped due to errors"
            break
        fi

        # Check if training completed
        if [ -f "$CHECKPOINT_DIR/training_complete" ]; then
            log_section "Training Completed Successfully!"
            break
        fi

        log "Restarting monitor loop..."
        sleep 5
    done

    log_section "Pipeline finished"
}

# Run main
main "$@"