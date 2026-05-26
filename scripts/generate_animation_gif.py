#!/usr/bin/env python3
"""
Generate animated GIF of P-EAGLE architecture.
Run this script to create an animated GIF that can be embedded in README.
"""

import os
import sys

# Use Pillow for creating GIF frames
from PIL import Image, ImageDraw, ImageFont

# Configuration
WIDTH = 900
HEIGHT = 700
BG_COLOR = (13, 17, 23)  # Dark background like GitHub dark mode
FONT_SIZE = 14
PADDING = 20

# Colors (RGB)
YELLOW = (255, 193, 7)
BLUE = (0, 136, 255)
GREEN = (40, 167, 69)
MAGENTA = (183, 71, 188)
CYAN = (23, 162, 184)
WHITE = (255, 255, 255)
DIM_GRAY = (139, 148, 158)
BORDER_COLOR = (48, 54, 61)


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def load_font(size=FONT_SIZE):
    """Load a font, fallback to default if not found."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/lucon.ttf",
    ]

    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue

    # Try default with size
    try:
        return ImageFont.load_default(size=size)
    except:
        try:
            return ImageFont.load_default()
        except:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def draw_box(draw, x, y, width, height, text_lines, border_color, fill_color=None, text_color=WHITE):
    """Draw a box with text."""
    # Fill
    if fill_color:
        draw.rectangle([x, y, x + width, y + height], fill=fill_color)

    # Border
    draw.rectangle([x, y, x + width, y + height], outline=border_color, width=2)

    # Text
    font = load_font()
    try:
        # Try new Pillow API
        line_height = font.getbbox("Test")[3] - font.getbbox("Test")[1] + 6
    except:
        try:
            # Old API
            line_height = font.getsize("Test")[1] + 6
        except:
            line_height = 18
    text_y = y + 10
    for line in text_lines:
        draw.text((x + 10, text_y), line, font=font, fill=text_color)
        text_y += line_height


def draw_arrow_down(draw, x, y, height, color, animated_parts=4):
    """Draw an animated arrow going down."""
    for i in range(animated_parts):
        arrow_y = y + i * (height // 4)
        draw.line([(x, y), (x, arrow_y + 5)], fill=color, width=2)
    draw.polygon([(x - 5, arrow_y + 5), (x + 5, arrow_y + 5), (x, arrow_y + 15)], fill=color)


def create_frame(components, title="P-EAGLE Architecture: EAGLE-3"):
    """Create a single frame with the given components visible."""
    # Create image
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    font_title = load_font(18)
    font = load_font(FONT_SIZE)
    font_small = load_font(12)

    # Title
    title_y = 20
    draw.text((WIDTH // 2 - 150, title_y), title, font=font_title, fill=WHITE)

    # Legend
    legend_items = [
        (YELLOW, "Target"),
        (BLUE, "Feature"),
        (GREEN, "EAGLE-3"),
        (WHITE, "Drafter"),
        (MAGENTA, "MTP"),
        (CYAN, "Tree"),
    ]

    legend_y = 55
    x_offset = 50
    for i, (color, name) in enumerate(legend_items):
        draw.text((x_offset + i * 130, legend_y), f"■ {name}", font=font_small, fill=color)

    # Draw components based on what's visible
    y_pos = 90

    # Target Model
    if "target" in components:
        box_width = 400
        box_height = 80
        box_x = (WIDTH - box_width) // 2
        draw_box(draw, box_x, y_pos, box_width, box_height,
                ["TARGET MODEL", "(e.g., Gemma-3-4B, Llama-3-8B)", "", "Hidden Dim: 2560 | Layers: 18-32"],
                YELLOW, YELLOW, WHITE)
        y_pos += box_height + 10

        # Arrow
        if "arrow1" in components:
            draw_arrow_down(draw, WIDTH // 2, y_pos, 30, YELLOW)

    # Feature Extractor
    if "feature" in components:
        y_pos += 35
        box_width = 350
        box_height = 70
        box_x = (WIDTH - box_width) // 2
        draw_box(draw, box_x, y_pos, box_width, box_height,
                ["FEATURE EXTRACTOR", "", "Extracts hidden states from target", "Output: (batch, seq, hidden)"],
                BLUE, BLUE, WHITE)
        y_pos += box_height + 10

        # Arrow
        if "arrow2" in components:
            draw_arrow_down(draw, WIDTH // 2, y_pos, 30, BLUE)

    # EAGLE-3 Injection
    if "eagle" in components:
        y_pos += 35
        box_width = 280
        box_height = 80
        box_x = (WIDTH - box_width) // 2
        draw_box(draw, box_x, y_pos, box_width, box_height,
                ["EAGLE-3 INJECTION", "", "[embeddings ⊕ hidden]", "First layer: 6656 → 640"],
                GREEN, GREEN, WHITE)
        y_pos += box_height + 10

        # Arrow
        if "arrow3" in components:
            draw_arrow_down(draw, WIDTH // 2, y_pos, 30, GREEN)

    # Drafter Model
    if "drafter" in components:
        y_pos += 35
        box_width = 500
        box_height = 140
        box_x = (WIDTH - box_width) // 2
        draw_box(draw, box_x, y_pos, box_width, box_height,
                ["DRAFTER MODEL (e.g., Gemma-3-270M)", "", "Hidden Dim: 640 | Layers: 6-12", "", "EAGLE-3 Layer 1: [embeddings ⊕ h] → 640",
                 "Standard Transformer Layers 2-N"],
                WHITE, None, WHITE)
        y_pos += box_height + 15

    # MTP Heads
    if "mtp" in components:
        head_width = 80
        head_height = 60
        spacing = 30
        start_x = (WIDTH - (4 * head_width + 3 * spacing)) // 2
        head_y = y_pos

        for i in range(4):
            hx = start_x + i * (head_width + spacing)
            draw_box(draw, hx, head_y, head_width, head_height,
                    [f"MTP Head {i+1}", "", f"h_{{t+{i+1}}}"],
                    MAGENTA, MAGENTA, WHITE)

        # Arrows from MTP heads
        if "arrow4" in components:
            center_x = WIDTH // 2
            draw.text((center_x - 60, head_y + head_height + 5), "K Parallel Tokens", font=font, fill=WHITE)

    # Tree Attention
    if "tree" in components:
        tree_y = head_y + head_height + 40
        box_width = 320
        box_height = 100
        box_x = (WIDTH - box_width) // 2
        draw_box(draw, box_x, tree_y, box_width, box_height,
                ["TREE ATTENTION", "", "Verify K tokens in 1 pass", "Accept/Reject via Target"],
                CYAN, CYAN, WHITE)

    # Footer
    draw.text((20, HEIGHT - 30), "P-EAGLE: 1.5-3x speedup through parallel speculative decoding", font=font_small, fill=DIM_GRAY)

    return img


def generate_animation_gif(output_path="docs/p-eagle-architecture.gif"):
    """Generate the animated GIF."""
    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    frames = []

    # Animation sequence - components appear one by one
    stages = [
        [],                                    # Empty frame
        ["target"],                            # Target model appears
        ["target", "arrow1"],
        ["target", "arrow1", "feature"],       # Feature extractor appears
        ["target", "arrow1", "feature", "arrow2"],
        ["target", "arrow1", "feature", "arrow2", "eagle"],  # EAGLE-3
        ["target", "arrow1", "feature", "arrow2", "eagle", "arrow3"],
        ["target", "arrow1", "feature", "arrow2", "eagle", "arrow3", "drafter"],  # Drafter
        ["target", "arrow1", "feature", "arrow2", "eagle", "arrow3", "drafter", "mtp"],  # MTP Heads
        ["target", "arrow1", "feature", "arrow2", "eagle", "arrow3", "drafter", "mtp", "arrow4"],
        ["target", "arrow1", "feature", "arrow2", "eagle", "arrow3", "drafter", "mtp", "arrow4", "tree"],  # Tree attention
    ]

    # Frame durations (in 100ths of a second)
    durations = [100, 500, 100, 800, 100, 500, 100, 800, 500, 100, 1000]

    print("Generating animation frames...")
    for i, components in enumerate(stages):
        frame = create_frame(components)
        frames.append(frame)
        print(f"  Frame {i+1}/{len(stages)}: {components if components else 'empty'}")

    print(f"\nSaving GIF to {output_path}...")
    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,  # Loop forever
        optimize=True
    )

    print(f"GIF saved successfully!")
    return output_path


def generate_workflow_gif(output_path="docs/p-eagle-workflow.gif"):
    """Generate the workflow animation GIF."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    frames = []
    font = load_font(FONT_SIZE)

    stages = [
        [],  # Empty
        ["raw"],  # Raw data
        ["raw", "processed"],  # Processed
        ["raw", "processed", "features"],  # Features
        ["raw", "processed", "features", "training"],  # Training
        ["raw", "processed", "features", "training", "inference"],  # Inference
        ["raw", "processed", "features", "training", "evaluate"],  # Evaluate
    ]

    durations = [100, 600, 600, 600, 600, 300, 1000]

    print("Generating workflow frames...")
    for i, components in enumerate(stages):
        img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)

        # Title
        font_title = load_font(18)
        draw.text((WIDTH // 2 - 150, 20), "Data Pipeline Workflow", font=font_title, fill=WHITE)

        # Draw pipeline boxes
        box_width = 120
        box_height = 60
        spacing = 40
        start_x = 80
        y_pos = 100

        labels = ["RAW", "PROCESSED", "FEATURES", "TRAINING", "CHECKPOINTS"]
        colors = [YELLOW, GREEN, BLUE, MAGENTA, CYAN]

        for j, (label, color) in enumerate(zip(labels, colors)):
            if label.lower() in components:
                bx = start_x + j * (box_width + spacing)
                draw.rectangle([bx, y_pos, bx + box_width, y_pos + box_height], fill=color, outline=color)
                draw.text((bx + 10, y_pos + 20), label, font=font, fill=WHITE if color != WHITE else BG_COLOR)

                # Arrow to next
                if j < 4 and labels[j+1].lower() in components:
                    ax = bx + box_width
                    draw.line([(ax, y_pos + box_height//2), (ax + spacing, y_pos + box_height//2)], fill=DIM_GRAY, width=2)

        # Show outputs
        if "training" in components or "inference" in components or "evaluate" in components:
            out_y = y_pos + box_height + 40
            draw.text((80, out_y), "Outputs: Inference (Speculative Decoding) | Evaluate (Metrics)", font=font, fill=DIM_GRAY)

        # Footer
        draw.text((20, HEIGHT - 30), "P-EAGLE Pipeline: data → features → training → inference", font=load_font(12), fill=DIM_GRAY)

        frames.append(img)
        print(f"  Frame {i+1}/{len(stages)}: {components if components else 'empty'}")

    print(f"\nSaving workflow GIF to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True
    )

    print(f"Workflow GIF saved successfully!")
    return output_path


if __name__ == "__main__":
    # Generate architecture animation
    arch_gif = generate_animation_gif()

    # Generate workflow animation
    workflow_gif = generate_workflow_gif()

    print("\n" + "="*60)
    print("GIFs generated successfully!")
    print(f"  Architecture: {arch_gif}")
    print(f"  Workflow: {workflow_gif}")
    print("="*60)
    print("\nAdd these to your README.md:")
    print(f'![P-EAGLE Architecture]({arch_gif})')
    print(f'![P-EAGLE Workflow]({workflow_gif})')