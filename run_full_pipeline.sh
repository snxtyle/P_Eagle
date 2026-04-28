#!/bin/bash
# P-EAGLE Full Pipeline Automation Script
# Universal version - works with any compatible model pair
#
# Usage:
#   ./run_full_pipeline.sh                          # Run with defaults
#   ./run_full_pipeline.sh --target <model> --drafter <model>  # Custom models
#   ./run_full_pipeline.sh --skip-data-gen          # Skip data generation
#
# Model Pair Compatibility:
#   - SAME FAMILY (RECOMMENDED): Gemma-7B + Gemma-2B, Qwen-7B + Qwen-1.5B
#   - DIFFERENT FAMILIES: May work but requires vocab alignment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================"
echo "P-EAGLE Full Pipeline"
echo "================================"

# ============================================================================
# DEFAULT CONFIGURATION (Change these or use command-line args)
# ============================================================================

# Option 1: Same-family models (RECOMMENDED - guaranteed vocab compatibility)
# TARGET_MODEL="google/gemma-2b-it"
# DRAFTER_MODEL="unsloth/gemma-2-2b"  # Same tokenizer as target

# Option 2: Qwen models (also same vocab)
# TARGET_MODEL="Qwen/Qwen2.5-7B-Instruct"
# DRAFTER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# DGX Sparx Configuration: Gemma 7B (target) + Gemma 3 270M (drafter)
TARGET_MODEL="google/gemma-7b-it"
DRAFTER_MODEL="google/gemma-3-270m-it"

# Dimensions
TARGET_HIDDEN_DIM="3072"   # gemma-7b: 3072
DRAFTER_HIDDEN_DIM="2048"  # gemma-3-270m: 2048

# Training parameters
SPECULATION_DEPTH="${SPECULATION_DEPTH:-6}"  # Increased from 4 - sequential position IDs make deeper heads accurate
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
LORA_RANK="${LORA_RANK:-64}"
QUANTIZATION="${QUANTIZATION:-8bit}"

# Paths
DATA_DIR="${DATA_DIR:-./data}"
FEATURES_DIR="$DATA_DIR/features"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
PROCESSED_DIR="$DATA_DIR/processed"
EVAL_OUTPUT="${EVAL_OUTPUT:-evaluation_results.json}"

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

SKIP_DATA_GEN=false
SKIP_FEATURE_EXTRACTION=false
SKIP_TRAINING=false
SKIP_EVALUATION=false
SKIP_COMPAT_CHECK=false
RUN_DRY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET_MODEL="$2"
            shift 2
            ;;
        --drafter)
            DRAFTER_MODEL="$2"
            shift 2
            ;;
        --target-hidden-dim)
            TARGET_HIDDEN_DIM="$2"
            shift 2
            ;;
        --skip-data-gen)
            SKIP_DATA_GEN=true
            shift
            ;;
        --skip-feature-extraction)
            SKIP_FEATURE_EXTRACTION=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --skip-compat-check)
            SKIP_COMPAT_CHECK=true
            shift
            ;;
        --dry-run)
            RUN_DRY=true
            shift
            ;;
        --help|-h)
            echo "P-EAGLE Full Pipeline"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Model Selection:"
            echo "  --target MODEL          Target model (default: $TARGET_MODEL)"
            echo "  --drafter MODEL         Drafter model (default: $DRAFTER_MODEL)"
            echo "  --target-hidden-dim N   Target hidden dimension (default: $TARGET_HIDDEN_DIM)"
            echo ""
            echo "Stage Control:"
            echo "  --skip-data-gen           Skip data generation"
            echo "  --skip-feature-extraction Skip feature extraction"
            echo "  --skip-training           Skip training"
            echo "  --skip-evaluation         Skip evaluation"
            echo "  --skip-compat-check       Skip compatibility check"
            echo "  --dry-run                 Show commands without executing"
            echo ""
            echo "Recommended Model Pairs:"
            echo "  Gemma: google/gemma-7b + unsloth/gemma-2-2b"
            echo "  Qwen:  Qwen/Qwen2.5-7B-Instruct + Qwen/Qwen2.5-1.5B-Instruct"
            echo ""
            echo "Examples:"
            echo "  $0                                              # Run all stages"
            echo "  $0 --skip-data-gen --skip-feature-extraction   # Retrain only"
            echo "  $0 --target Qwen/Qwen2.5-7B-Instruct --drafter Qwen/Qwen2.5-1.5B-Instruct"
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use --help for usage)"
            exit 1
            ;;
    esac
done

# ============================================================================
# STEP 0: Environment Setup
# ============================================================================
echo ""
echo "Step 0: Environment Setup"
echo "-------------------------"

mkdir -p "$DATA_DIR"{"/raw","/processed","/features","/output"}
mkdir -p "$CHECKPOINT_DIR" "$OUTPUT_DIR" ./logs ./plot_scripts/plots

# Check Python
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" || {
    echo "ERROR: PyTorch not available"
    exit 1
}

# Print configuration
echo ""
echo "Configuration:"
echo "  Target Model:   $TARGET_MODEL"
echo "  Drafter Model:  $DRAFTER_MODEL"
echo "  Hidden Dim:     $TARGET_HIDDEN_DIM"
echo "  Speculation K:  $SPECULATION_DEPTH"
echo "  Epochs:         $EPOCHS"
echo "  Batch Size:     $BATCH_SIZE"
echo "  Learning Rate:  $LEARNING_RATE"
echo "  LoRA Rank:      $LORA_RANK"

if [ "$RUN_DRY" = true ]; then
    echo ""
    echo "*** DRY RUN MODE - Commands will be shown but not executed ***"
fi

# ============================================================================
# STEP 0.5: Model Compatibility Check
# ============================================================================
if [ "$SKIP_COMPAT_CHECK" = false ]; then
    echo ""
    echo "Step 0.5: Model Compatibility Check"
    echo "-----------------------------------"

    if [ "$RUN_DRY" = false ]; then
        python3 check_model_compatibility.py --target "$TARGET_MODEL" --drafter "$DRAFTER_MODEL" || {
            echo ""
            echo "WARNING: Model compatibility check failed!"
            echo ""
            echo "Options:"
            echo "  1. Use models from the same family (e.g., both Gemma or both Qwen)"
            echo "  2. Continue anyway (may produce gibberish output)"
            echo "  3. Run with --skip-compat-check to skip this check"
            echo ""
            read -p "Continue anyway? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Aborted. Run: python3 check_model_compatibility.py --recommend"
                exit 1
            fi
        }
    else
        echo "Would run: python3 check_model_compatibility.py --target $TARGET_MODEL --drafter $DRAFTER_MODEL"
    fi
fi

# ============================================================================
# STEP 1: Data Preparation
# ============================================================================
if [ "$SKIP_DATA_GEN" = false ]; then
    echo ""
    echo "Step 1: Data Preparation"
    echo "------------------------"

    # Check for processed data
    PROCESSED_COUNT=$(find "$PROCESSED_DIR" -name "*.json" -type f 2>/dev/null | wc -l)

    if [ "$PROCESSED_COUNT" -eq 0 ]; then
        echo "ERROR: No processed JSON files found in $PROCESSED_DIR"
        echo "Please populate $PROCESSED_DIR with conversation JSON files"
        exit 1
    fi

    echo "Found $PROCESSED_COUNT processed files"

    CMD="python3 scripts/generate_data.py --local --num-samples $NUM_SAMPLES --input-dir $PROCESSED_DIR --output $DATA_DIR/output --format openai --output-format jsonl --min-words 30 --deduplicate"

    if [ "$RUN_DRY" = true ]; then
        echo "CMD: $CMD"
    else
        eval $CMD
    fi

    # Find the generated dataset file
    DATASET_FILE=$(find "$DATA_DIR/output" -name "dataset_*.jsonl" -type f 2>/dev/null | sort -t_ -k2,2n -k3 | tail -1)

    if [ -z "$DATASET_FILE" ] && [ "$RUN_DRY" = false ]; then
        echo "ERROR: No dataset file generated"
        exit 1
    fi

    echo "Dataset: $DATASET_FILE"
else
    echo "Skipping data generation"
    DATASET_FILE=$(find "$DATA_DIR/output" -name "dataset_*.jsonl" -type f 2>/dev/null | sort -t_ -k2,2n -k3 | tail -1)
    echo "Using existing: $DATASET_FILE"
fi

# ============================================================================
# STEP 2: Feature Extraction
# ============================================================================
if [ "$SKIP_FEATURE_EXTRACTION" = false ]; then
    echo ""
    echo "Step 2: Feature Extraction"
    echo "--------------------------"

    # Clear old features
    if [ "$RUN_DRY" = false ]; then
        rm -f "$FEATURES_DIR"/*.pt
    fi

    # IMPORTANT: Use drafter's tokenizer to ensure training compatibility
    CMD="python3 -m p_eagle.scripts.extract_features \
        --model_path $TARGET_MODEL \
        --tokenizer_path $DRAFTER_MODEL \
        --input_data $DATASET_FILE \
        --output_dir $FEATURES_DIR \
        --quantization $QUANTIZATION \
        --layers early,middle,final \
        --fusion mean \
        --batch_size 2 \
        --shard_size 5000 \
        --max_length 2048"

    if [ "$RUN_DRY" = true ]; then
        echo "CMD: $CMD"
    else
        eval $CMD
    fi

    if [ "$RUN_DRY" = false ]; then
        FEATURE_COUNT=$(find "$FEATURES_DIR" -name "*.pt" -type f | wc -l)
        echo "Extracted $FEATURE_COUNT feature shards"
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

    # Clear old checkpoints except logs
    if [ "$RUN_DRY" = false ]; then
        mkdir -p "$CHECKPOINT_DIR"
        find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 ! -name "logs" -type d -exec rm -rf {} + 2>/dev/null || true
    fi

    CMD="python3 -m p_eagle.scripts.train_drafter \
        --drafter_model $DRAFTER_MODEL \
        --target_hidden_dim $TARGET_HIDDEN_DIM \
        --feature_dir $FEATURES_DIR \
        --output_dir $CHECKPOINT_DIR \
        --num_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --warmup_steps 100 \
        --speculation_depth $SPECULATION_DEPTH \
        --use_lora \
        --lora_rank $LORA_RANK \
        --skip-hardware-check"

    if [ "$RUN_DRY" = true ]; then
        echo "CMD: $CMD"
    else
        eval $CMD
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

    CMD="python3 -m p_eagle.scripts.evaluate \
        --drafter_checkpoint $CHECKPOINT_DIR/best_model \
        --target_model $TARGET_MODEL \
        --baseline \
        --max_tokens 100 \
        --domain_test \
        --output $EVAL_OUTPUT"

    if [ "$RUN_DRY" = true ]; then
        echo "CMD: $CMD"
    else
        eval $CMD
        echo "Results: $EVAL_OUTPUT"
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

CMD="python3 -m plot_scripts.generate_plots --mode all --checkpoint_dirs $CHECKPOINT_DIR --eval_file $EVAL_OUTPUT --output_dir plot_scripts/plots"

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
echo "Quick Commands:"
echo "  Test inference:"
echo "    python3 -m p_eagle.scripts.run_inference \\"
echo "      --target_model $TARGET_MODEL \\"
echo "      --drafter_checkpoint $CHECKPOINT_DIR/best_model \\"
echo "      --prompt 'Your prompt here'"
echo ""
echo "  Check model compatibility:"
echo "    python3 check_model_compatibility.py --target <model> --drafter <model>"
echo ""
