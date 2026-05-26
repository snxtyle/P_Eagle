#!/usr/bin/env python3
"""
P-EAGLE Architecture & Workflow Animation

Run this script to see an animated visualization of:
1. The EAGLE-3 Architecture
2. The Data Pipeline Workflow
3. The Speculative Decoding Process

Usage: python scripts/animate_workflow.py [--arch | --workflow | --all]
"""

import sys
import time

# Colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Foreground colors
YELLOW = "\033[33m"
GREEN = "\033[32m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
RED = "\033[31m"

# Bright colors
BRIGHT_BLUE = "\033[94m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_WHITE = "\033[97m"


# ============================================================
# TIMING CONFIGURATION - Adjust these for faster/slower animation
# ============================================================
DELAY_CHAR = 0.01       # Typewriter effect delay per character
DELAY_LINE = 0.3         # Delay between lines
DELAY_SECTION = 0.8      # Delay between major sections
DELAY_PAUSE = 1.5        # Longer pause for important content
DELAY_ARROW = 0.15       # Arrow animation speed

def clear_screen():
    """Clear the screen and move cursor to top-left."""
    print("\033[2J\033[H", end="")
    sys.stdout.flush()


def hide_cursor():
    """Hide the terminal cursor."""
    print("\033[?25l", end="")
    sys.stdout.flush()


def show_cursor():
    """Show the terminal cursor."""
    print("\033[?25h", end="")
    sys.stdout.flush()


def pause(seconds=None):
    """Pause between animations. Use default if no seconds specified."""
    time.sleep(seconds if seconds else DELAY_SECTION)


def type_text(text, color=WHITE, delay=None):
    """Type text character by character for emphasis."""
    d = delay if delay else DELAY_CHAR
    for char in text:
        print(f"{color}{char}{RESET}", end="", flush=True)
        time.sleep(d)
    print()


def print_line(text, color=WHITE, bold=False):
    """Print a single line with optional styling."""
    prefix = f"{BOLD}" if bold else ""
    print(f"{prefix}{color}{text}{RESET}")


def print_box_line(text, width, color, align="center"):
    """Print a line that fits inside a box."""
    if align == "center":
        padding = max(0, (width - len(text) - 2) // 2)
        print(f"{color}║{RESET}{' ' * padding}{BOLD}{text}{RESET}{' ' * (width - 2 - padding - len(text))} {color}║{RESET}")
    elif align == "left":
        print(f"{color}║{RESET} {BOLD}{text}{RESET}{' ' * (width - len(text) - 3)} {color}║{RESET}")


def animate_down_arrow(color=WHITE, width=30):
    """Show an animated down arrow."""
    for i in range(4):
        spaces = " " * width
        arrows = "│" * (i + 1)
        print(f"{color}{spaces}{arrows}{RESET}\r", end="", flush=True)
        time.sleep(DELAY_ARROW)
    print(f"{color}{' ' * width}▼{RESET}")


# ============================================================
# ARCHITECTURE ANIMATION
# ============================================================

def show_architecture():
    """Display architecture diagram with readable pacing."""
    clear_screen()
    hide_cursor()

    print()
    print_line("╔══════════════════════════════════════════════════════════════════════════════╗", BRIGHT_CYAN, bold=True)
    print_line("║                     P-EAGLE ARCHITECTURE: EAGLE-3                           ║", BRIGHT_CYAN, bold=True)
    print_line("╚══════════════════════════════════════════════════════════════════════════════╝", BRIGHT_CYAN, bold=True)
    print()

    pause()

    # ----- TARGET MODEL -----
    print_line("┌────────────────────────────────────────────────────────────────────────────┐", YELLOW)
    print_line("│                           TARGET MODEL                                       │", YELLOW, bold=True)
    print_line("│                      (e.g., Gemma-3-4B, Llama-3-8B)                         │", DIM)
    print_line("│                                                                            │", YELLOW)
    print_line("│   Purpose: Main language model providing hidden states for the drafter    │", WHITE)
    print_line("│   Output:  Hidden states from last N layers (e.g., layers -1, -2, -3)     │", WHITE)
    print_line("│   Hidden:  2560 dims (4B model) or 4096 dims (8B model)                    │", DIM)
    print_line("└────────────────────────────────────────────────────────────────────────────┘", YELLOW)

    pause(DELAY_PAUSE)

    animate_down_arrow(YELLOW)

    # ----- FEATURE EXTRACTOR -----
    print()
    print_line("┌────────────────────────────────────────────────────────────────────────────┐", BRIGHT_BLUE)
    print_line("│                         FEATURE EXTRACTOR                                  │", BRIGHT_BLUE, bold=True)
    print_line("│                                                                            │", BRIGHT_BLUE)
    print_line("│   Extracts hidden states from the target model                            │", WHITE)
    print_line("│   Combines multiple layers using mean/concat fusion                       │", WHITE)
    print_line("│   Output:  Hidden state tensor (batch, seq_len, hidden_dim)              │", DIM)
    print_line("└────────────────────────────────────────────────────────────────────────────┘", BRIGHT_BLUE)

    pause(DELAY_PAUSE)

    animate_down_arrow(BRIGHT_BLUE)

    # ----- EAGLE-3 INJECTION -----
    print()
    print_line("                             ╔════════════════════════════╗", BRIGHT_GREEN)
    print_line("                             ║      EAGLE-3 INJECTION      ║", BRIGHT_GREEN, bold=True)
    print_line("                             ║                            ║", BRIGHT_GREEN)
    print_line('                             ║  [embeddings ⊕ hidden]     ║', WHITE)
    print_line("                             ║   Concatenate to create     ║", WHITE)
    print_line("                             ║   first layer input         ║", WHITE)
    print_line("                             ╚════════════════════════════╝", BRIGHT_GREEN)

    pause(DELAY_PAUSE)

    animate_down_arrow(BRIGHT_GREEN)

    # ----- DRAFTER MODEL -----
    print()
    print_line("┌────────────────────────────────────────────────────────────────────────────┐", GREEN)
    print_line("│                            DRAFTER MODEL                                   │", GREEN, bold=True)
    print_line("│                       (e.g., Gemma-3-270M, Llama-3-1B)                     │", DIM)
    print_line("│                                                                            │", GREEN)
    print_line("│   Architecture: Lightweight transformer with EAGLE-3 first layer           │", WHITE)
    print_line("│   Hidden Dim:  640 dims (270M model) - much smaller than target          │", WHITE)
    print_line("│   Layers:      ~6-12 layers vs 18-32 in target                           │", DIM)
    print_line("│                                                                            │", GREEN)

    print_line("│                        ┌───────────────────────────┐                       │", GREEN)
    print_line("│                        │  EAGLE-3 Layer 1          │                       │", WHITE)
    print_line("│                        │  Input: embeddings ⊕ h    │                       │", DIM)
    print_line("│                        │  Output: 640-dim states   │                       │", DIM)
    print_line("│                        └───────────────────────────┘                       │", GREEN)

    print_line("│                        ┌───────────────────────────┐                       │", GREEN)
    print_line("│                        │  Standard Transformer     │                       │", WHITE)
    print_line("│                        │  Layers 2-N              │                       │", DIM)
    print_line("│                        └───────────────────────────┘                       │", GREEN)

    print_line("│                                                                            │", GREEN)
    print_line("└────────────────────────────────────────────────────────────────────────────┘", GREEN)

    pause(DELAY_PAUSE)

    # ----- MTP HEADS -----
    print()
    print_line("              ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐", MAGENTA)
    print_line("              │  MTP Head   │ │  MTP Head   │ │  MTP Head   │ │  MTP Head   │", MAGENTA)
    print_line("              │      1      │ │      2      │ │      3      │ │      K      │", WHITE)
    print_line("              │ predicts    │ │ predicts    │ │ predicts    │ │ predicts    │", DIM)
    print_line("              │   h_{{t+1}}    │ │   h_{{t+2}}    │ │   h_{{t+3}}    │ │   h_{{t+K}}    │", DIM)
    print_line("              └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘", MAGENTA)
    print_line("                     └───────────────┴───────────────┴───────────────┘", WHITE)
    print_line("                                          │", WHITE)
    print_line("                                          ▼", WHITE)
    print_line("                              ┌─────────────────────────┐", WHITE)
    print_line("                              │  K Parallel Tokens      │", WHITE, bold=True)
    print_line("                              │  (single forward pass)  │", DIM)
    print_line("                              └─────────────────────────┘", WHITE)

    pause(DELAY_PAUSE)

    animate_down_arrow(MAGENTA)

    # ----- TREE ATTENTION -----
    print()
    print_line("                         ┌─────────────────────────────────┐", CYAN)
    print_line("                         │         TREE ATTENTION          │", CYAN, bold=True)
    print_line("                         │                                 │", CYAN)
    print_line("                         │  Verifies ALL K tokens in ONE    │", WHITE)
    print_line("                         │  forward pass through target     │", WHITE)
    print_line("                         │                                 │", CYAN)
    print_line("                         │  1. Draft matches target?        │", WHITE)
    print_line("                         │     → ACCEPT (use drafted token) │", GREEN)
    print_line("                         │                                 │", CYAN)
    print_line("                         │  2. Draft differs from target?   │", WHITE)
    print_line("                         │     → REJECT + RESAMPLE         │", RED)
    print_line("                         │                                 │", CYAN)
    print_line("                         │  3. After first rejection,       │", WHITE)
    print_line("                         │     all subsequent rejected      │", DIM)
    print_line("                         │                                 │", CYAN)
    print_line("                         └─────────────────────────────────┘", CYAN)
    print()

    pause(DELAY_PAUSE)

    # ----- LEGEND -----
    print_line("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", WHITE, bold=True)
    print_line("  LEGEND:", WHITE, bold=True)
    print_line(f"  {YELLOW}■{RESET} Yellow  = Target Model      (4B-8B) Main LLM providing hidden states", WHITE)
    print_line(f"  {BRIGHT_BLUE}■{RESET} Blue    = Feature Extractor (Extracts hidden states from target)", WHITE)
    print_line(f"  {GREEN}■{RESET} Green   = EAGLE-3 Injection  (Key innovation: concat embeddings + hidden)", WHITE)
    print_line(f"  {WHITE}■{RESET} White   = Drafter Model      (270M-1B) Lightweight draft generator", WHITE)
    print_line(f"  {MAGENTA}■{RESET} Magenta = MTP Heads         (Multi-Token Prediction heads)", WHITE)
    print_line(f"  {CYAN}■{RESET} Cyan    = Tree Attention     (Parallel verification of K tokens)", WHITE)
    print_line("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", WHITE, bold=True)
    print()

    show_cursor()


# ============================================================
# WORKFLOW ANIMATION
# ============================================================

def show_workflow():
    """Display workflow diagram with readable pacing."""
    clear_screen()
    hide_cursor()

    print()
    print_line("╔══════════════════════════════════════════════════════════════════════════════╗", BRIGHT_CYAN, bold=True)
    print_line("║                            DATA PIPELINE WORKFLOW                            ║", BRIGHT_CYAN, bold=True)
    print_line("╚══════════════════════════════════════════════════════════════════════════════╝", BRIGHT_CYAN, bold=True)
    print()

    pause()

    # STEP 1: RAW DATA
    print_line("╔═══════════════════════════════════════════════════════════════════════════════╗", YELLOW)
    print_line("║                                                                               ║", YELLOW)
    print_line("║   STEP 1: RAW DATA                                                            ║", YELLOW, bold=True)
    print_line("║                                                                               ║", YELLOW)
    print_line("║   ┌─────────────────────────────────────────────────────────────────────┐    ║", YELLOW)
    print_line("║   │  Input Format:                                                           │    ║", WHITE)
    print_line("║   │      JSONL with fields: text, system, messages                          │    ║", DIM)
    print_line("║   │                                                                          │    ║", YELLOW)
    print_line("║   │  Sources:                                                                 │    ║", WHITE)
    print_line("║   │      - Instruction datasets (Alpaca, Vicuna format)                    │    ║", DIM)
    print_line("║   │      - Conversation data                                                 │    ║", DIM)
    print_line("║   │      - Synthetic data generated by LLMs                                 │    ║", DIM)
    print_line("║   │                                                                          │    ║", YELLOW)
    print_line("║   │  Initial filtering:                                                      │    ║", WHITE)
    print_line("║   │      - Remove empty/duplicate entries                                  │    ║", DIM)
    print_line("║   │      - Quality checks                                                    │    ║", DIM)
    print_line("║   └─────────────────────────────────────────────────────────────────────┘    ║", YELLOW)
    print_line("║                                                                               ║", YELLOW)
    print_line("╚═══════════════════════════════════════════════════════════════════════════════╝", YELLOW)
    print()

    pause(DELAY_PAUSE)
    animate_down_arrow(WHITE)

    # STEP 2: GENERATE DATA
    print_line("╔═══════════════════════════════════════════════════════════════════════════════╗", GREEN)
    print_line("║                                                                               ║", GREEN)
    print_line("║   STEP 2: PROCESS DATA (generate_data.py)                                    ║", GREEN, bold=True)
    print_line("║                                                                               ║", GREEN)
    print_line("║   ┌─────────────────────────────────────────────────────────────────────┐    ║", GREEN)
    print_line("║   │  python scripts/generate_data.py --local --num-samples 5000            │    ║", DIM)
    print_line("║   │                                                                          │    ║", GREEN)
    print_line("║   │  Operations:                                                             │    ║", WHITE)
    print_line("║   │      1. Tokenize with target model's tokenizer                          │    ║", DIM)
    print_line("║   │      2. Filter by length (min: 256, max: 4096 tokens)                   │    ║", DIM)
    print_line("║   │      3. Deduplicate similar sequences                                   │    ║", DIM)
    print_line("║   │      4. Format for training (input/target pairs)                        │    ║", DIM)
    print_line("║   │                                                                          │    ║", GREEN)
    print_line("║   │  Output: data/processed/*.jsonl                                           │    ║", WHITE)
    print_line("║   └─────────────────────────────────────────────────────────────────────┘    ║", GREEN)
    print_line("║                                                                               ║", GREEN)
    print_line("╚═══════════════════════════════════════════════════════════════════════════════╝", GREEN)
    print()

    pause(DELAY_PAUSE)
    animate_down_arrow(WHITE)

    # STEP 3: EXTRACT FEATURES
    print_line("╔═══════════════════════════════════════════════════════════════════════════════╗", BRIGHT_BLUE)
    print_line("║                                                                               ║", BRIGHT_BLUE)
    print_line("║   STEP 3: EXTRACT FEATURES (feature_extractor.py)                             ║", BRIGHT_BLUE, bold=True)
    print_line("║                                                                               ║", BRIGHT_BLUE)
    print_line("║   ┌─────────────────────────────────────────────────────────────────────┐    ║", BRIGHT_BLUE)
    print_line("║   │  python -m p_eagle.scripts.extract_features                            │    ║", DIM)
    print_line("║   │                                                                          │    ║", BRIGHT_BLUE)
    print_line("║   │  Operations:                                                             │    ║", WHITE)
    print_line("║   │      1. Run target model forward pass (no gradient)                     │    ║", DIM)
    print_line("║   │      2. Extract hidden states from specified layers                      │    ║", DIM)
    print_line("║   │      3. Fuse layers (mean or concat)                                    │    ║", DIM)
    print_line("║   │      4. Save as .pt tensor shards                                       │    ║", DIM)
    print_line("║   │                                                                          │    ║", BRIGHT_BLUE)
    print_line("║   │  Example Layers: --layers -1,-2,-3                                       │    ║", WHITE)
    print_line("║   │  Fusion Methods: --fusion mean | concat                                  │    ║", DIM)
    print_line("║   │                                                                          │    ║", BRIGHT_BLUE)
    print_line("║   │  Output: data/features/shard_*.pt                                         │    ║", WHITE)
    print_line("║   └─────────────────────────────────────────────────────────────────────┘    ║", BRIGHT_BLUE)
    print_line("║                                                                               ║", BRIGHT_BLUE)
    print_line("╚═══════════════════════════════════════════════════════════════════════════════╝", BRIGHT_BLUE)
    print()

    pause(DELAY_PAUSE)
    animate_down_arrow(WHITE)

    # STEP 4: TRAIN
    print_line("╔═══════════════════════════════════════════════════════════════════════════════╗", MAGENTA)
    print_line("║                                                                               ║", MAGENTA)
    print_line("║   STEP 4: TRAIN DRAFTER (train_drafter.py / EagleTrainer)                     ║", MAGENTA, bold=True)
    print_line("║                                                                               ║", MAGENTA)
    print_line("║   ┌─────────────────────────────────────────────────────────────────────┐    ║", MAGENTA)
    print_line("║   │  ./automation.sh single  OR  python -m p_eagle.scripts.train_drafter │    ║", DIM)
    print_line("║   │                                                                          │    ║", MAGENTA)
    print_line("║   │  Architecture:                                                            │    ║", WHITE)
    print_line("║   │      1. Load base model (e.g., Gemma-3-270M)                            │    ║", DIM)
    print_line("║   │      2. Add EAGLE-3 first layer (4096 → 640 dim projection)              │    ║", DIM)
    print_line("║   │      3. Add K MTP heads (each predicts next token at position +i)        │    ║", DIM)
    print_line("║   │      4. Apply LoRA adaptation (rank=64, alpha=128)                        │    ║", DIM)
    print_line("║   │                                                                          │    ║", MAGENTA)
    print_line("║   │  Loss Function:                                                           │    ║", WHITE)
    print_line("║   │      KL divergence between drafter and target token distributions        │    ║", DIM)
    print_line("║   │                                                                          │    ║", MAGENTA)
    print_line("║   │  Output: checkpoints/best_model/                                          │    ║", WHITE)
    print_line("║   │      - model.pt (trained weights)                                         │    ║", DIM)
    print_line("║   │      - config.json (architecture config)                                 │    ║", DIM)
    print_line("║   └─────────────────────────────────────────────────────────────────────┘    ║", MAGENTA)
    print_line("║                                                                               ║", MAGENTA)
    print_line("╚═══════════════════════════════════════════════════════════════════════════════╝", MAGENTA)
    print()

    pause(DELAY_PAUSE)
    animate_down_arrow(WHITE)

    # STEP 5: EVALUATE
    print_line("╔═══════════════════════════════════════════════════════════════════════════════╗", CYAN)
    print_line("║                                                                               ║", CYAN)
    print_line("║   STEP 5: EVALUATE & INFERENCE                                                ║", CYAN, bold=True)
    print_line("║                                                                               ║", CYAN)
    print_line("║   ┌─────────────────────────────────────────────────────────────────────┐    ║", CYAN)
    print_line("║   │  python -m p_eagle.scripts.evaluate                                    │    ║", DIM)
    print_line("║   │                                                                          │    ║", CYAN)
    print_line("║   │  Metrics:                                                                │    ║", WHITE)
    print_line("║   │      - Acceptance Rate: % of drafted tokens accepted (>70% target)     │    ║", DIM)
    print_line("║   │      - Speedup: Tokens/second vs autoregressive baseline (1.5-3x)      │    ║", DIM)
    print_line("║   │      - Quality: Perplexity match with target model                     │    ║", DIM)
    print_line("║   │                                                                          │    ║", CYAN)
    print_line("║   │  Inference:                                                              │    ║", WHITE)
    print_line("║   │      - Speculative decoding with Tree Attention                        │    ║", DIM)
    print_line("║   │      - Batch processing support                                         │    ║", DIM)
    print_line("║   └─────────────────────────────────────────────────────────────────────┘    ║", CYAN)
    print_line("║                                                                               ║", CYAN)
    print_line("╚═══════════════════════════════════════════════════════════════════════════════╝", CYAN)
    print()

    pause(DELAY_PAUSE)

    # Quick Commands
    print_line("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", WHITE, bold=True)
    print_line("  QUICK COMMANDS:", WHITE, bold=True)
    print_line("  ./run_full_pipeline.sh              - Run complete pipeline end-to-end", GREEN)
    print_line("  python scripts/generate_data.py     - Step 1: Generate processed data", YELLOW)
    print_line("  python -m p_eagle.scripts.extract_features  - Step 2: Extract hidden states", BRIGHT_BLUE)
    print_line("  ./automation.sh single             - Step 3: Train drafter (single GPU)", MAGENTA)
    print_line("  python -m p_eagle.scripts.evaluate - Step 4: Evaluate & benchmark", CYAN)
    print_line("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", WHITE, bold=True)
    print()

    show_cursor()


# ============================================================
# SPECULATIVE DECODING ANIMATION
# ============================================================

def show_speculative_decoding():
    """Show speculative decoding process with readable pacing."""
    clear_screen()
    hide_cursor()

    print()
    print_line("╔══════════════════════════════════════════════════════════════════════════════╗", BRIGHT_MAGENTA, bold=True)
    print_line("║                       SPECULATIVE DECODING PROCESS                             ║", BRIGHT_MAGENTA, bold=True)
    print_line("╚══════════════════════════════════════════════════════════════════════════════╝", BRIGHT_MAGENTA, bold=True)
    print()

    pause()

    # STEP 1: GENERATE DRAFT
    print_line("╔═════════════════════════════════════════════════════════════════════════════════╗", YELLOW)
    print_line("║                                                                                 ║", YELLOW)
    print_line("║   STEP 1: DRAFTER GENERATES K SPECULATIVE TOKENS                                ║", YELLOW, bold=True)
    print_line("║                                                                                 ║", YELLOW)
    print_line("║   ┌───────────────────────────────────────────────────────────────────────────┐ ║", YELLOW)
    print_line("║   │                                                                               │ ║", YELLOW)
    print_line("║   │   Input to Drafter:                                                           │ ║", WHITE)
    print_line("║   │       - Token embeddings from context                                         │ ║", DIM)
    print_line("║   │       - Hidden states from target model (EAGLE-3 injection)                   │ ║", DIM)
    print_line("║   │                                                                               │ ║", YELLOW)
    print_line("║   │   Drafter Processing:                                                        │ ║", WHITE)
    print_line("║   │       - EAGLE-3 first layer: [embeddings ⊕ hidden] → 640 dims                 │ ║", DIM)
    print_line("║   │       - Standard transformer layers 2-N                                     │ ║", DIM)
    print_line("║   │       - K MTP heads each predict next token                                  │ ║", DIM)
    print_line("║   │                                                                               │ ║", YELLOW)
    print_line("║   │   Output: K tokens in PARALLEL (single forward pass)                           │ ║", WHITE)
    print_line("║   │                                                                               │ ║", YELLOW)
    print_line("║   │   Example (K=4):                                                             │ ║", DIM)
    print_line("║   │       Draft: [the] [cat] [sat] [on]                                          │ ║", WHITE)
    print_line("║   │                                                                               │ ║", YELLOW)
    print_line("║   └───────────────────────────────────────────────────────────────────────────┘ ║", YELLOW)
    print_line("║                                                                                 ║", YELLOW)
    print_line("╚═════════════════════════════════════════════════════════════════════════════════╝", YELLOW)
    print()

    pause(DELAY_PAUSE)
    animate_down_arrow(WHITE)

    # STEP 2: VERIFY
    print_line("╔═════════════════════════════════════════════════════════════════════════════════╗", GREEN)
    print_line("║                                                                                 ║", GREEN)
    print_line("║   STEP 2: TREE ATTENTION VERIFIES ALL K TOKENS                                  ║", GREEN, bold=True)
    print_line("║                                                                                 ║", GREEN)
    print_line("║   ┌───────────────────────────────────────────────────────────────────────────┐ ║", GREEN)
    print_line("║   │                                                                               │ ║", GREEN)
    print_line("║   │   Target Model Processing:                                                   │ ║", WHITE)
    print_line("║   │       - Single forward pass with ALL K drafted tokens                       │ ║", DIM)
    print_line("║   │       - Tree attention mask allows parallel verification                     │ ║", DIM)
    print_line("║   │       - No sequential dependency between tokens                              │ ║", DIM)
    print_line("║   │                                                                               │ ║", GREEN)
    print_line("║   │   Example:                                                                   │ ║", DIM)
    print_line("║   │                                                                               │ ║", GREEN)
    print_line("║   │       Drafted:  [the]  [cat]  [sat]  [on]                                    │ ║", WHITE)
    print_line("║   │       Target:   [the]  [cat]  [sat]  [mat]  ◄── Different!                  │ ║", WHITE)
    print_line("║   │       Match:      ✓      ✓      ✓      ✗                                   │ ║", DIM)
    print_line("║   │                                                                               │ ║", GREEN)
    print_line("║   │   Key Insight:                                                               │ ║", DIM)
    print_line("║   │       Tree attention verifies K tokens in time of 1 autoregressive step      │ ║", DIM)
    print_line("║   │                                                                               │ ║", GREEN)
    print_line("║   └───────────────────────────────────────────────────────────────────────────┘ ║", GREEN)
    print_line("║                                                                                 ║", GREEN)
    print_line("╚═════════════════════════════════════════════════════════════════════════════════╝", GREEN)
    print()

    pause(DELAY_PAUSE)
    animate_down_arrow(WHITE)

    # STEP 3: ACCEPT/REJECT
    print_line("╔═════════════════════════════════════════════════════════════════════════════════╗", BRIGHT_BLUE)
    print_line("║                                                                                 ║", BRIGHT_BLUE)
    print_line("║   STEP 3: ACCEPT / REJECT DECISIONS                                             ║", BRIGHT_BLUE, bold=True)
    print_line("║                                                                                 ║", BRIGHT_BLUE)
    print_line("║   ┌───────────────────────────────────────────────────────────────────────────┐ ║", BRIGHT_BLUE)
    print_line("║   │                                                                               │ ║", BRIGHT_BLUE)
    print_line("║   │   Decision Rules:                                                            │ ║", WHITE)
    print_line("║   │                                                                               │ ║", BRIGHT_BLUE)
    print_line("║   │       1. Token MATCHES target prediction                                     │ ║", WHITE)
    print_line("║   │          → ACCEPT the drafted token (use it in output)                      │ ║", GREEN)
    print_line("║   │                                                                               │ ║", BRIGHT_BLUE)
    print_line("║   │       2. Token DIFFERS from target prediction                               │ ║", WHITE)
    print_line("║   │          → REJECT and RESAMPLE using target distribution                    │ ║", RED)
    print_line("║   │                                                                               │ ║", BRIGHT_BLUE)
    print_line("║   │       3. After FIRST rejection                                               │ ║", WHITE)
    print_line("║   │          → ALL subsequent tokens also REJECTED                              │ ║", RED)
    print_line("║   │                                                                               │ ║", BRIGHT_BLUE)
    print_line("║   │   Example:                                                                   │ ║", DIM)
    print_line("║   │       Drafted:  [the]  [cat]  [sat]  [on]                                    │ ║", WHITE)
    print_line("║   │       Result:    [the]✓ [cat]✓ [sat]✓ [on]✗                                 │ ║", WHITE)
    print_line("║   │                                                                               │ ║", BRIGHT_BLUE)
    print_line("║   │       Outcome:  Accept 3 tokens, resample 1 token                            │ ║", WHITE)
    print_line("║   │                                                                               │ ║", BRIGHT_BLUE)
    print_line("║   │   SPEEDUP: 3 tokens decoded in time of 1 autoregressive step!               │ ║", GREEN, bold=True)
    print_line("║   │                                                                               │ ║", BRIGHT_BLUE)
    print_line("║   └───────────────────────────────────────────────────────────────────────────┘ ║", BRIGHT_BLUE)
    print_line("║                                                                                 ║", BRIGHT_BLUE)
    print_line("╚═════════════════════════════════════════════════════════════════════════════════╝", BRIGHT_BLUE)
    print()

    pause(DELAY_PAUSE)

    # KEY METRICS
    print_line("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", WHITE, bold=True)
    print_line("  KEY METRICS:", WHITE, bold=True)
    print_line("                                                                                 ", WHITE)
    print_line("  ┌─────────────────────────────────────────────────────────────────────────────┐", WHITE)
    print_line("  │  Metric            │  Target Value  │  Notes                                  │", WHITE)
    print_line("  │  ─────────────────│───────────────│─────────────────────────────────────────│", DIM)
    print_line("  │  Acceptance Rate  │     >70%      │  % of drafted tokens accepted         │", WHITE)
    print_line("  │  Speedup         │    1.5-3x     │  Tokens/second vs autoregressive       │", WHITE)
    print_line("  │  Memory Overhead  │     +270MB    │  Drafter model + KV cache               │", DIM)
    print_line("  │  Output Quality   │   Identical   │  Probabilistically same as target       │", DIM)
    print_line("  └─────────────────────────────────────────────────────────────────────────────┘", WHITE)
    print()

    show_cursor()


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="P-EAGLE Architecture Animation")
    parser.add_argument("--arch", action="store_true", help="Show architecture diagram")
    parser.add_argument("--workflow", action="store_true", help="Show workflow diagram")
    parser.add_argument("--speculative", action="store_true", help="Show speculative decoding")
    parser.add_argument("--all", action="store_true", help="Show all animations")
    args = parser.parse_args()

    if not any([args.arch, args.workflow, args.speculative, args.all]):
        args.all = True

    try:
        if args.all or args.arch:
            show_architecture()
            if not args.all:
                input(f"\n{BOLD}{DIM}Press Enter to exit...{RESET}")

        if args.all or args.workflow:
            if args.all:
                pause(1)
            show_workflow()
            if not args.all:
                input(f"\n{BOLD}{DIM}Press Enter to exit...{RESET}")

        if args.all or args.speculative:
            if args.all:
                pause(1)
            show_speculative_decoding()

    except KeyboardInterrupt:
        clear_screen()
        show_cursor()
        print(f"\n{BOLD}{YELLOW}Animation interrupted.{RESET}")
        sys.exit(0)

    clear_screen()
    show_cursor()


if __name__ == "__main__":
    main()