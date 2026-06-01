#!/usr/bin/env python3
"""
Convert "new data" folder to single JSONL file in OpenAI format.

Reads all JSON files in the input directory, filters for valid OpenAI
format conversations (with messages array containing role/content),
and writes them to a single JSONL file.

Usage:
    python scripts/convert_to_openai_format.py \
        --input "/path/to/new data" \
        --output data/openai_format.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm


def is_valid_openai_message(msg):
    """Check if a message dict has valid role/content format."""
    if not isinstance(msg, dict):
        return False
    if "role" not in msg:
        return False
    role = msg.get("role", "")
    if not isinstance(role, str):
        return False
    if role not in ("system", "user", "assistant", "tool", "function"):
        return False
    # Content can be string, list, or None
    return True


def extract_messages(data):
    """
    Extract messages from various JSON structures.

    Handles:
    - Direct messages array: [{"role": ..., "content": ...}]
    - Nested in 'data', 'messages', 'conversation' keys
    - LiteLLM log format
    """
    # Case 1: Direct list of messages
    if isinstance(data, list):
        # Check if it's a list of message dicts
        if data and isinstance(data[0], dict) and "role" in data[0]:
            return data
        # Check if it's a list with potential message content inside objects
        for item in data:
            if isinstance(item, dict):
                # Try nested extraction
                result = extract_messages(item)
                if result:
                    return result

    # Case 2: Dict with messages directly
    if isinstance(data, dict):
        # Direct messages key
        if "messages" in data and isinstance(data["messages"], list):
            msgs = data["messages"]
            if msgs and isinstance(msgs[0], dict) and "role" in msgs[0]:
                return msgs

        # LiteLLM format: data.messages
        if "data" in data and isinstance(data["data"], dict):
            return extract_messages(data["data"])

        # Conversation wrapper
        if "conversation" in data and isinstance(data["conversation"], list):
            return data["conversation"]

        # Choices format (OpenAI API response style)
        if "choices" in data and isinstance(data["choices"], list):
            for choice in data["choices"]:
                if isinstance(choice, dict) and "message" in choice:
                    return [choice["message"]]

        # Try first element if data is a list wrapper
        for key in ["data", "items", "results", "conversations"]:
            if key in data and isinstance(data[key], list):
                result = extract_messages({key: data[key]})
                if result:
                    return result

    return None


def parse_content(content):
    """
    Parse content field which can be:
    - String
    - List of {type: "text", text: "..."}
    - List of various formats
    - None
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                # Text content
                if "text" in item:
                    parts.append(item["text"])
                elif "content" in item:
                    parts.append(str(item["content"]))
                elif item.get("type") == "text":
                    parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def convert_to_openai_format(messages):
    """Convert messages to clean OpenAI format."""
    converted = []
    for msg in messages:
        if not is_valid_openai_message(msg):
            continue

        role = msg.get("role", "")
        content = parse_content(msg.get("content"))

        # Skip empty content (but keep system messages)
        if not content and role not in ("system",):
            continue

        new_msg = {
            "role": role,
            "content": content
        }

        # Preserve tool_calls if present
        if "tool_calls" in msg:
            new_msg["tool_calls"] = msg["tool_calls"]

        # Preserve name for tool/function messages
        if "name" in msg:
            new_msg["name"] = msg["name"]

        converted.append(new_msg)

    return converted


def process_file(json_path):
    """Process a single JSON file and return OpenAI format if valid."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract messages
        messages = extract_messages(data)
        if not messages:
            return None

        # Check if we have at least one valid message
        has_valid = any(is_valid_openai_message(m) for m in messages)
        if not has_valid:
            return None

        # Convert to OpenAI format
        converted = convert_to_openai_format(messages)
        if not converted:
            return None

        # Return in OpenAI format
        return {"messages": converted}

    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert to OpenAI format JSONL")
    parser.add_argument("--input", required=True, help="Input directory with JSON files")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    # Get all JSON files
    json_files = sorted(input_dir.glob("*.json"))
    if args.limit:
        json_files = json_files[:args.limit]

    print(f"Found {len(json_files)} JSON files in {input_dir}")

    # Process and write
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    invalid_count = 0

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for json_file in tqdm(json_files, desc="Processing files"):
            result = process_file(json_file)

            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                valid_count += 1
            else:
                invalid_count += 1

    print(f"\nComplete!")
    print(f"  Valid conversations: {valid_count}")
    print(f"  Skipped (no valid format): {invalid_count}")
    print(f"  Output: {output_path}")

    # Print file size
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()