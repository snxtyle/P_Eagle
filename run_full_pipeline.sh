#!/bin/bash
# P-EAGLE Full Pipeline Automation Script
# Proven working configuration based on production fixes
#
# Usage:
#   ./run_full_pipeline.sh                              # Run all stages (single GPU)
#   ./run_full_pipeline.sh multi <master_ip>           # Multi-GPU DDP mode
#   ./run_full_pipeline.sh deepspeed <master_ip>       # DeepSpeed ZeRO-3 mode
#   ./run_full_pipeline.sh --skip-data-gen --skip-feature-extraction  # Train only
#   SPECULATION_DEPTH=1 LORA_RANK=16 ./run_full_pipeline.sh          # Override defaults
#   ./run_full_pipeline.sh --input-jsonl data/my_data.jsonl          # Use custom data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# PARSE MODE ARGUMENT (must be first for multi/deepspeed modes)
# ============================================================================
MODE="${1:-single}"  # single, multi, or deepspeed
MASTER_IP="${2:-}"

if [ "$MODE" = "multi" ] || [ "$MODE" = "deepspeed" ]; then
    if [ -z "$MASTER_IP" ]; then
        echo "ERROR: Master IP required for $MODE mode"
        echo "Usage: $0 $MODE <master_ip> [options...]"
        exit 1
    fi
    shift 2  # Remove mode args, keep remaining args for parse loop
else
    MODE="single"
fi

echo "================================"
echo "P-EAGLE Full Pipeline"
echo "Mode: $MODE"
echo "================================"

# ============================================================================
# DEFAULT CONFIGURATION (proven working values)
# Override any variable via environment, e.g.: LORA_RANK=32 EPOCHS=3 ./run_full_pipeline.sh
# ============================================================================

# Models
TARGET_MODEL="${TARGET_MODEL:-google/gemma-3-4b-it}"
DRAFTER_MODEL="${DRAFTER_MODEL:-google/gemma-3-270m-it}"
TARGET_HIDDEN_DIM="${TARGET_HIDDEN_DIM:-2560}"   # gemma-3-4b: 2560
DRAFTER_HIDDEN_DIM="${DRAFTER_HIDDEN_DIM:-640}"  # gemma-3-270m: 640

# Training parameters (optimized for quality and stability)
SPECULATION_DEPTH="${SPECULATION_DEPTH:-4}"       # 4 MTP heads for better speculation
NUM_SAMPLES="${NUM_SAMPLES:-2000}"                # Synthetic data samples (if generating)
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-3}"                             # Full training
LEARNING_RATE="${LEARNING_RATE:-5e-5}"            # Optimal for LoRA
WARMUP_STEPS="${WARMUP_STEPS:-100}"
LORA_RANK="${LORA_RANK:-32}"                      # Higher rank for better quality
LORA_ALPHA="${LORA_ALPHA:-128}"                   # 4x rank for better scaling
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-8}"  # Better GPU utilization
QUANTIZATION="${QUANTIZATION:-}"                  # Empty = no quantization (avoids NaN issues)
USE_LORA="${USE_LORA:-true}"
USE_FLASH_ATTENTION="${USE_FLASH_ATTENTION:-true}"

# Regularization parameters
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
MTP_DROPOUT="${MTP_DROPOUT:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

# Feature extraction parameters
FEAT_BATCH_SIZE="${FEAT_BATCH_SIZE:-1}"           # Small batch to avoid OOM on target model
FEAT_SHARD_SIZE="${FEAT_SHARD_SIZE:-100}"         # ~3-4GB shards (was 500, caused OOM)
FEAT_MAX_LENGTH="${FEAT_MAX_LENGTH:-2048}"        # Was 4096, reduced for memory
FEAT_LAYERS="${FEAT_LAYERS:-last}"
FEAT_FUSION="${FEAT_FUSION:-mean}"

# Paths
DATA_DIR="${DATA_DIR:-./data}"
FEATURES_DIR="$DATA_DIR/features"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

# ============================================================================
# MULTI-GPU / DEEPSPEED CONFIGURATION
# ============================================================================

# DeepSpeed ZeRO-3 configuration (JSON string, written to temp file at runtime)
DEEPSPEED_CONFIG='{
  "train_batch_size": 2,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {"lr": 5e-05}
  },
  "fp16": {"enabled": false},
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}'

# Set output directory based on mode
if [ "$MODE" = "single" ]; then
    OUTPUT_DIR="${OUTPUT_DIR:-./output}"
elif [ "$MODE" = "multi" ]; then
    OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/automated_multi}"
elif [ "$MODE" = "deepspeed" ]; then
    OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/deepspeed_run}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
PROCESSED_DIR="$DATA_DIR/processed"
EVAL_OUTPUT="${EVAL_OUTPUT:-evaluation_results.json}"
LOGS_DIR="${LOGS_DIR:-./logs}"
INPUT_JSONL="${INPUT_JSONL:-}"                    # Direct path to pre-existing JSONL data

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

SKIP_DATA_GEN=false
SKIP_FEATURE_EXTRACTION=false
SKIP_TRAINING=false
SKIP_EVALUATION=false
SKIP_SECURITY_CHECK=false
RUN_DRY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET_MODEL="$2"; shift 2 ;;
        --drafter)
            DRAFTER_MODEL="$2"; shift 2 ;;
        --target-hidden-dim)
            TARGET_HIDDEN_DIM="$2"; shift 2 ;;
        --speculation-depth)
            SPECULATION_DEPTH="$2"; shift 2 ;;
        --lora-rank)
            LORA_RANK="$2"; shift 2 ;;
        --lora-alpha)
            LORA_ALPHA="$2"; shift 2 ;;
        --epochs)
            EPOCHS="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --gradient-accumulation)
            GRADIENT_ACCUMULATION="$2"; shift 2 ;;
        --learning-rate)
            LEARNING_RATE="$2"; shift 2 ;;
        --warmup-steps)
            WARMUP_STEPS="$2"; shift 2 ;;
        --shard-size)
            FEAT_SHARD_SIZE="$2"; shift 2 ;;
        --max-length)
            FEAT_MAX_LENGTH="$2"; shift 2 ;;
        --input-jsonl)
            INPUT_JSONL="$2"; shift 2 ;;
        --skip-data-gen)
            SKIP_DATA_GEN=true; shift ;;
        --skip-feature-extraction)
            SKIP_FEATURE_EXTRACTION=true; shift ;;
        --skip-training)
            SKIP_TRAINING=true; shift ;;
        --skip-evaluation)
            SKIP_EVALUATION=true; shift ;;
        --dry-run)
            RUN_DRY=true; shift ;;
        --skip-security-check)
            SKIP_SECURITY_CHECK=true; shift ;;
        --help|-h)
            cat << 'HELP_EOF'
P-EAGLE Full Pipeline

Usage:
  ./run_full_pipeline.sh [OPTIONS]                          # Single GPU mode
  ./run_full_pipeline.sh multi <master_ip> [OPTIONS]       # Multi-GPU DDP
  ./run_full_pipeline.sh deepspeed <master_ip> [OPTIONS]  # DeepSpeed ZeRO-3

Model Selection:
  --target MODEL             Target model (default: google/gemma-3-4b-it)
  --drafter MODEL            Drafter model (default: google/gemma-3-270m-it)
  --target-hidden-dim N      Target hidden dimension (default: 2560)

Training Parameters:
  --speculation-depth K      Number of MTP heads (default: 4)
  --lora-rank R              LoRA rank (default: 32)
  --lora-alpha A             LoRA alpha scaling (default: 128)
  --epochs N                 Training epochs (default: 3)
  --batch-size N             Batch size (default: 4)
  --gradient-accumulation N  Gradient accumulation steps (default: 8)
  --learning-rate LR         Learning rate (default: 5e-5)
  --warmup-steps N           Warmup steps (default: 100)

Feature Extraction:
  --shard-size N             Samples per shard (default: 100)
  --max-length N             Max sequence length (default: 2048)
  --input-jsonl PATH         Use existing JSONL instead of generating data

Stage Control:
  --skip-data-gen            Skip data generation
  --skip-feature-extraction  Skip feature extraction
  --skip-training            Skip training
  --skip-evaluation          Skip evaluation
  --dry-run                  Show commands without executing

Environment Variables (override defaults):
  TARGET_MODEL, DRAFTER_MODEL, SPECULATION_DEPTH, LORA_RANK, LORA_ALPHA,
  EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION, LEARNING_RATE, WARMUP_STEPS,
  FEAT_SHARD_SIZE, FEAT_MAX_LENGTH, QUANTIZATION, INPUT_JSONL

Examples:
  ./run_full_pipeline.sh                                     # Single GPU
  ./run_full_pipeline.sh multi 192.168.1.100                 # Multi-GPU
  ./run_full_pipeline.sh deepspeed 192.168.1.100             # DeepSpeed
  ./run_full_pipeline.sh --skip-data-gen --skip-feature-extraction
  ./run_full_pipeline.sh --input-jsonl data/raw/converted_train.jsonl
  SPECULATION_DEPTH=2 EPOCHS=3 ./run_full_pipeline.sh
HELP_EOF
            exit 0 ;;
        *)
            echo "Unknown option: $1 (use --help for usage)"
            exit 1 ;;
    esac
done

# ============================================================================
# STEP 0: Environment Setup
# ============================================================================
echo ""
echo "Step 0: Environment Setup"
echo "-------------------------"

mkdir -p "$DATA_DIR"{"/raw","/processed","/features","/output"}
mkdir -p "$CHECKPOINT_DIR" "$OUTPUT_DIR" "$LOGS_DIR" ./plot_scripts/plots

# Use venv Python if available
if [ -z "$PYTHON_CMD" ]; then
    if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
        PYTHON_CMD="$SCRIPT_DIR/venv/bin/python"
    elif [ -n "$VIRTUAL_ENV" ]; then
        PYTHON_CMD="${VIRTUAL_ENV}/bin/python"
    else
        PYTHON_CMD="python3"
    fi
fi

$PYTHON_CMD --version
$PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" || {
    echo "ERROR: PyTorch not available"
    exit 1
}

# Print configuration
echo ""
echo "Configuration:"
echo "  Target Model:          $TARGET_MODEL"
echo "  Drafter Model:         $DRAFTER_MODEL"
echo "  Hidden Dim:            $TARGET_HIDDEN_DIM"
echo "  Speculation K:         $SPECULATION_DEPTH"
echo "  Epochs:                $EPOCHS"
echo "  Batch Size:            $BATCH_SIZE"
echo "  Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "  Learning Rate:        $LEARNING_RATE"
echo "  Warmup Steps:          $WARMUP_STEPS"
echo "  LoRA Rank:             $LORA_RANK"
echo "  LoRA Alpha:            $LORA_ALPHA"
echo "  Quantization:          ${QUANTIZATION:-none}"
echo "  Shard Size:            $FEAT_SHARD_SIZE"
echo "  Max Length:            $FEAT_MAX_LENGTH"
echo ""
echo "  Regularization:"
echo "    Label Smoothing: $LABEL_SMOOTHING"
echo "    MTP Dropout:     $MTP_DROPOUT"
echo "    Weight Decay:    $WEIGHT_DECAY"

if [ "$RUN_DRY" = true ]; then
    echo ""
    echo "*** DRY RUN MODE - Commands will be shown but not executed ***"
fi

# ============================================================================
# STEP 1: Data Preparation
# ============================================================================
if [ "$SKIP_DATA_GEN" = false ]; then
    echo ""
    echo "Step 1: Data Preparation"
    echo "------------------------"

    # If input JSONL is provided directly, skip generation
    if [ -n "$INPUT_JSONL" ] && [ -f "$INPUT_JSONL" ]; then
        echo "Using provided JSONL: $INPUT_JSONL"
        DATASET_FILE="$INPUT_JSONL"
    else
        # FIRST: Check for combined_dataset.jsonl (the 7k+ sample dataset)
        COMBINED_FILE="$DATA_DIR/output/combined_dataset.jsonl"
        if [ -f "$COMBINED_FILE" ]; then
            echo "Found combined_dataset.jsonl: $COMBINED_FILE"
            DATASET_FILE="$COMBINED_FILE"
        else
            # Check for processed data (jsonl format)
            PROCESSED_COUNT=$(find "$PROCESSED_DIR" -name "*.jsonl" -type f 2>/dev/null | wc -l)

            if [ "$PROCESSED_COUNT" -eq 0 ]; then
                echo "ERROR: No processed JSONL files found in $PROCESSED_DIR"
                echo "Provide --input-jsonl PATH or populate $PROCESSED_DIR"
                exit 1
            fi

            echo "Found $PROCESSED_COUNT processed files"

            CMD="$PYTHON_CMD scripts/generate_data.py --local --num-samples $NUM_SAMPLES --input-dir $PROCESSED_DIR --output $DATA_DIR/output --format openai --output-format jsonl --deduplicate"

            if [ "$RUN_DRY" = true ]; then
                echo "CMD: $CMD"
            else
                DATA_LOG="$LOGS_DIR/data_gen_$(date +%Y%m%d_%H%M%S).log"
                echo "Data generation started. Logging to: $DATA_LOG"
                eval $CMD 2>&1 | tee "$DATA_LOG"
            fi

            DATASET_FILE=$(find "$DATA_DIR/output" -name "dataset_*.jsonl" -type f 2>/dev/null | sort -t_ -k2,2n -k3 | tail -1)

            if [ -z "$DATASET_FILE" ] && [ "$RUN_DRY" = false ]; then
                echo "ERROR: No dataset file generated"
                exit 1
            fi

            echo "Dataset: $DATASET_FILE"
        fi
    fi
else
    echo "Skipping data generation"
    if [ -n "$INPUT_JSONL" ] && [ -f "$INPUT_JSONL" ]; then
        DATASET_FILE="$INPUT_JSONL"
    else
        # FIRST: Check for combined_dataset.jsonl
        COMBINED_FILE="$DATA_DIR/output/combined_dataset.jsonl"
        if [ -f "$COMBINED_FILE" ]; then
            DATASET_FILE="$COMBINED_FILE"
        else
            DATASET_FILE=$(find "$DATA_DIR/output" -name "dataset_*.jsonl" -type f 2>/dev/null | sort -t_ -k2,2n -k3 | tail -1)
        fi
    fi
    echo "Using existing: $DATASET_FILE"
fi

# ============================================================================
# STEP 2: Feature Extraction
# ============================================================================
if [ "$SKIP_FEATURE_EXTRACTION" = false ]; then
    echo ""
    echo "Step 2: Feature Extraction"
    echo "--------------------------"

    if [ -z "$DATASET_FILE" ]; then
        echo "ERROR: No dataset file available for feature extraction"
        echo "Run without --skip-data-gen or provide --input-jsonl"
        exit 1
    fi

    # Clear old features
    if [ "$RUN_DRY" = false ]; then
        rm -f "$FEATURES_DIR"/*.pt
    fi

    CMD="$PYTHON_CMD -m p_eagle.scripts.extract_features \
        --model_path $TARGET_MODEL \
        --tokenizer_path $DRAFTER_MODEL \
        --input_data $DATASET_FILE \
        --output_dir $FEATURES_DIR \
        --layers $FEAT_LAYERS \
        --fusion $FEAT_FUSION \
        --batch_size $FEAT_BATCH_SIZE \
        --shard_size $FEAT_SHARD_SIZE \
        --max_length $FEAT_MAX_LENGTH"

    if [ "$RUN_DRY" = true ]; then
        echo "CMD: $CMD"
    else
        FEAT_LOG="$LOGS_DIR/feature_extraction_$(date +%Y%m%d_%H%M%S).log"
        echo "Feature extraction started. Logging to: $FEAT_LOG"
        eval $CMD 2>&1 | tee "$FEAT_LOG"
    fi

    if [ "$RUN_DRY" = false ]; then
        FEATURE_COUNT=$(find "$FEATURES_DIR" -name "*.pt" -type f | wc -l)
        echo "Extracted $FEATURE_COUNT feature shards"

        # Verify shard sizes
        echo ""
        echo "Verifying feature shards..."
        $PYTHON_CMD -c "
import glob, os
pts = glob.glob('$FEATURES_DIR/*_shard*.pt')
if pts:
    sizes = [os.path.getsize(p)/1e9 for p in pts]
    print(f'  Shards: {len(pts)}')
    print(f'  Size range: {min(sizes):.1f}GB - {max(sizes):.1f}GB')
    print(f'  Total: {sum(sizes):.1f}GB')
    if max(sizes) > 10:
        print(f'  WARNING: Some shards exceed 10GB and may cause OOM during training!')
        print(f'  Consider reducing --shard-size')
"
    fi
else
    echo "Skipping feature extraction"
fi

# ============================================================================
# STEP 3: Training
# ============================================================================
if [ "$SKIP_TRAINING" = false ]; then
    echo ""
    echo "Step 3: Training Drafter"
    echo "------------------------"
    echo "Mode: $MODE"

    # Clear old checkpoints except logs
    if [ "$RUN_DRY" = false ]; then
        mkdir -p "$CHECKPOINT_DIR"
        find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 ! -name "logs" -type d -exec rm -rf {} + 2>/dev/null || true
    fi

    echo "Setting up training optimizations..."
    export PYTHONDONTWRITEBYTECODE=1
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    # Build training command with optional flags
    TRAIN_FLAGS=""
    [ "$USE_LORA" = "true" ] && TRAIN_FLAGS="$TRAIN_FLAGS --use_lora"
    [ "$USE_FLASH_ATTENTION" = "true" ] && TRAIN_FLAGS="$TRAIN_FLAGS --use_flash_attention"
    [ -n "$QUANTIZATION" ] && TRAIN_FLAGS="$TRAIN_FLAGS --quantization $QUANTIZATION"

    # Base training args
    BASE_TRAIN_ARGS=(
        --drafter_model "$DRAFTER_MODEL"
        --target_hidden_dim "$TARGET_HIDDEN_DIM"
        --feature_dir "$FEATURES_DIR"
        --output_dir "$CHECKPOINT_DIR"
        --num_epochs "$EPOCHS"
        --batch_size "$BATCH_SIZE"
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION"
        --learning_rate "$LEARNING_RATE"
        --warmup_steps "$WARMUP_STEPS"
        --speculation_depth "$SPECULATION_DEPTH"
        --lora_rank "$LORA_RANK"
        --lora_alpha "$LORA_ALPHA"
        --label_smoothing "$LABEL_SMOOTHING"
        --mtp_dropout "$MTP_DROPOUT"
        --weight_decay "$WEIGHT_DECAY"
        --save_every 250
        $TRAIN_FLAGS
        --yes
    )

    if [ "$MODE" = "single" ]; then
        # Single GPU training
        CMD="$PYTHON_CMD -m p_eagle.scripts.train_drafter ${BASE_TRAIN_ARGS[*]}"

    elif [ "$MODE" = "deepspeed" ]; then
        echo "Setting up DeepSpeed ZeRO-3 configuration..."
        # Create DeepSpeed config file
        echo "$DEEPSPEED_CONFIG" > /tmp/ds_config.json

        # Create hostfile for multi-node
        echo "$MASTER_IP slots=1" > /tmp/ds_hostfile
        echo "DeepSpeed config created at /tmp/ds_config.json"

        # DeepSpeed launch command
        CMD="deepspeed \
            --hostfile=/tmp/ds_hostfile \
            --master_addr=$MASTER_IP \
            --master_port=29500 \
            --module p_eagle.training.trainer \
            ${BASE_TRAIN_ARGS[*]} \
            --deepspeed /tmp/ds_config.json"

    else
        # Multi-GPU DDP training
        echo "Setting up Multi-GPU DDP training..."
        # Determine rank based on IP
        MY_IP=$(hostname -I | awk '{print $1}')
        if [[ "$MY_IP" == "$MASTER_IP" ]]; then
            RANK=0
            NODE_RANK=0
            echo "Detected as Master (Rank 0)"
        else
            RANK=1
            NODE_RANK=1
            echo "Detected as Worker (Rank 1)"
        fi

        export MASTER_ADDR="$MASTER_IP"
        export MASTER_PORT=29500
        export WORLD_SIZE=2
        export RANK="$RANK"

        CMD="torchrun \
            --nnodes=2 \
            --nproc_per_node=1 \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            -m p_eagle.training.trainer \
            ${BASE_TRAIN_ARGS[*]}"
    fi

    if [ "$RUN_DRY" = true ]; then
        echo "CMD: $CMD"
    else
        TRAIN_LOG="$LOGS_DIR/training_$(date +%Y%m%d_%H%M%S).log"
        echo "Training started. Logging to: $TRAIN_LOG"
        echo "View live: tail -f $TRAIN_LOG"
        eval $CMD 2>&1 | tee "$TRAIN_LOG"
        echo "Training complete. Best model: $CHECKPOINT_DIR/best_model"
    fi
else
    echo "Skipping training"
fi

# ============================================================================
# STEP 4: Evaluation
# ============================================================================
if [ "$SKIP_EVALUATION" = false ]; then
    echo ""
    echo "Step 4: Evaluation"
    echo "------------------"

    BEST_MODEL="$CHECKPOINT_DIR/best_model"
    if [ ! -d "$BEST_MODEL" ]; then
        echo "WARNING: Best model not found at $BEST_MODEL, skipping evaluation"
    else
        CMD="$PYTHON_CMD -m p_eagle.scripts.evaluate \
            --drafter_checkpoint $BEST_MODEL \
            --target_model $TARGET_MODEL \
            --baseline \
            --max_tokens 100 \
            --domain_test \
            --output $EVAL_OUTPUT"

        if [ "$RUN_DRY" = true ]; then
            echo "CMD: $CMD"
        else
            EVAL_LOG="$LOGS_DIR/evaluation_$(date +%Y%m%d_%H%M%S).log"
            echo "Evaluation started. Logging to: $EVAL_LOG"
            eval $CMD 2>&1 | tee "$EVAL_LOG"
            echo "Results: $EVAL_OUTPUT"

            # Display key metrics
            if [ -f "$EVAL_OUTPUT" ]; then
                echo ""
                echo "Key Metrics:"
                $PYTHON_CMD -c "
import json
with open('$EVAL_OUTPUT', 'r') as f:
    data = json.load(f)
    if 'mean_acceptance_length' in data:
        print(f\"  Mean Acceptance Length (MAL): {data['mean_acceptance_length']:.2f}\")
    if 'speedup' in data:
        print(f\"  Speedup: {data['speedup']:.2f}x\")
"
            fi
        fi
    fi
else
    echo "Skipping evaluation"
fi

# ============================================================================
# STEP 5: Plotting
# ============================================================================
echo ""
echo "Step 5: Generating Plots"
echo "------------------------"

CMD="$PYTHON_CMD -m plot_scripts.generate_plots --mode all --checkpoint_dirs $CHECKPOINT_DIR --eval_file $EVAL_OUTPUT --output_dir plot_scripts/plots"

if [ "$RUN_DRY" = true ]; then
    echo "CMD: $CMD"
else
    eval $CMD 2>/dev/null || echo "Plotting skipped (may require matplotlib)"
fi

# ============================================================================
# Done
# ============================================================================
echo ""
echo "================================"
if [ "$RUN_DRY" = true ]; then
    echo "DRY RUN COMPLETE"
else
    echo "PIPELINE COMPLETE!"
fi
echo "================================"
echo ""
echo "Summary:"
echo "  Mode: $MODE"
if [ "$MODE" != "single" ]; then
    echo "  Master IP: $MASTER_IP"
fi
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Logs directory: $LOGS_DIR"
echo "Log files:"
ls -1t "$LOGS_DIR"/*.log 2>/dev/null | head -10 | while read f; do echo "  - $f"; done
echo ""
echo "Quick Commands:"
echo "  Test inference:"
echo "    $PYTHON_CMD -m p_eagle.scripts.run_inference \\"
echo "      --target_model $TARGET_MODEL \\"
echo "      --drafter_checkpoint $CHECKPOINT_DIR/best_model \\"
echo "      --prompt 'Your prompt here'"
echo ""
