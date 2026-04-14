import json
import os
import re
import argparse
from glob import glob
from typing import Optional, List, Dict, Any, Tuple

# Default paths
DEFAULT_INPUT_DIR = "/Users/suraj.nagre/Desktop/juspay-eval-multilingual/processed"
DEFAULT_OUTPUT_FILE = "/Users/suraj.nagre/Desktop/juspay-eval-multilingual/output/openai_dataset.jsonl"

# HuggingFace dataset config
HF_DATASET_NAME = "Salesforce/xlam-function-calling-60k"

# Try to import datasets, handle gracefully if not installed
try:
    from datasets import load_dataset, Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets library not installed. Run: pip install datasets")

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv
    # Load .env file if exists
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", os.getenv("HF_TOKEN"))
    if HF_TOKEN:
        print(f"✓ HuggingFace token loaded from .env")
    else:
        HF_TOKEN = None
except ImportError:
    HF_TOKEN = None
    print("Note: python-dotenv not installed. Create .env file for HF token.")


def parse_tool_calls(answers_str: str) -> List[Dict[str, Any]]:
    """Parse tool calls from the answers field."""
    import ast
    try:
        # Try to parse as JSON first
        tool_calls = json.loads(answers_str)
        if isinstance(tool_calls, list):
            return tool_calls
    except:
        pass
    
    # Try parsing as string representation of list
    try:
        tool_calls = ast.literal_eval(answers_str)
        if isinstance(tool_calls, list):
            return tool_calls
    except:
        pass
    
    return []


def convert_xlam_to_messages(item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """Convert xlam dataset item to OpenAI messages format with tool_calls."""
    query = item.get("query", "")
    answers = item.get("answers", "")
    tools = item.get("tools", "")
    
    if not query or not answers:
        return None
    
    # Parse tool calls from answers
    tool_calls_data = parse_tool_calls(answers)
    if not tool_calls_data:
        return None
    
    # Parse tools if available
    tools_list = []
    try:
        tools_list = json.loads(tools) if tools else []
    except:
        pass
    
    # Build the conversation in OpenAI format
    messages = [
        {"role": "user", "content": query}
    ]
    
    # Add tool_calls to assistant message
    assistant_tool_calls = []
    for tc in tool_calls_data:
        name = tc.get("name", "")
        arguments = tc.get("arguments", {})
        
        # Convert arguments to JSON string if it's a dict
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        
        assistant_tool_calls.append({
            "id": f"call_{idx}_{name}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments
            }
        })
    
    # Add assistant message with tool_calls
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": assistant_tool_calls
    })
    
    return {
        "messages": messages,
        "_source": "huggingface",
        "_dataset": HF_DATASET_NAME,
        "_index": idx
    }


def has_tool_calls(messages: List[Dict[str, Any]]) -> bool:
    """Check if any assistant message in the conversation has tool_calls."""
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return True
    return False


def load_from_huggingface(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load data from Salesforce/xlam-function-calling-60k HuggingFace dataset.
    
    Converts to OpenAI format with tool_calls.
    
    Note: This dataset is gated and requires HuggingFace authentication.
    The token is automatically loaded from HUGGINGFACE_TOKEN or HF_TOKEN in .env file.
    """
    if not HF_AVAILABLE:
        print("Error: datasets library not available")
        return []
    
    print(f"Loading dataset: {HF_DATASET_NAME}")
    
    # Check for token
    if not HF_TOKEN:
        print("Warning: No HuggingFace token found in .env file.")
        print("Set HUGGINGFACE_TOKEN or HF_TOKEN in .env for gated datasets.")
        print("Get token from: https://huggingface.co/settings/tokens")
    
    try:
        # Load the dataset with token if available
        dataset = load_dataset(HF_DATASET_NAME, token=HF_TOKEN if HF_TOKEN else False)
        
        # Get the split (usually 'train')
        if isinstance(dataset, dict):
            # Multiple splits - use first available
            split = list(dataset.keys())[0]
            dataset = dataset[split]
        
        print(f"Total samples in dataset: {len(dataset)}")
        
        # Convert all samples to messages format
        valid_samples = []
        for idx, item in enumerate(dataset):
            sample = convert_xlam_to_messages(item, idx)
            
            if sample:
                valid_samples.append(sample)
                
                if limit and len(valid_samples) >= limit:
                    break
        
        print(f"Samples with tool_calls: {len(valid_samples)}")
        return valid_samples
        
    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "authentication" in error_msg.lower():
            print("Error: This dataset requires HuggingFace authentication.")
            print("To fix:")
            print("  1. Get a token from: https://huggingface.co/settings/tokens")
            print("  2. Run: huggingface-cli login")
            print("  3. Or add token to the code: load_dataset(..., token='your_token')")
        else:
            print(f"Error loading HuggingFace dataset: {e}")
        return []


def load_from_local(input_dir: str) -> List[Dict[str, Any]]:
    """Load data from local JSON files in the processed directory."""
    files = glob(os.path.join(input_dir, "*.json"))
    
    print(f"Loading {len(files)} files from {input_dir}")
    
    valid_samples = []
    for file_path in files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Get messages
            if isinstance(data, list):
                messages = data
            elif isinstance(data, dict):
                messages = data.get("messages", [])
            else:
                continue
            
            # Check if has tool_calls
            if has_tool_calls(messages):
                sample = {
                    "messages": messages,
                    "_source": "local",
                    "_file": os.path.basename(file_path)
                }
                valid_samples.append(sample)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Local samples with tool_calls: {len(valid_samples)}")
    return valid_samples


# 🔥 Clean assistant content
def clean_content(content):
    if not content or not isinstance(content, str):
        return None

    content = content.strip()

    # 1. remove ```json ... ```
    content = re.sub(r"```json\s*(.*?)\s*```", r"\1", content, flags=re.DOTALL)

    # 2. remove ``` ... ```
    content = re.sub(r"```\s*(.*?)\s*```", r"\1", content, flags=re.DOTALL)

    content = content.strip()

    # 3. try parsing JSON string content
    try:
        parsed = json.loads(content)

        # case: list of structured blocks
        if isinstance(parsed, list):
            texts = []
            for item in parsed:
                if isinstance(item, dict):
                    if "text" in item:
                        texts.append(item["text"])
                    elif "value" in item:
                        texts.append(item["value"])
            if texts:
                return "\n".join(texts)

        # case: dict → convert to string (clean JSON)
        if isinstance(parsed, dict):
            return json.dumps(parsed, indent=2)

    except:
        pass

    return content


def extract_clean_sample(data, file_path=None):
    # 🔥 Handle formats
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict):
        messages = data.get("messages", [])
    else:
        return None, "invalid_format"

    # nested list
    if len(messages) > 0 and isinstance(messages[0], list):
        messages = messages[0]

    system_msg = None
    user_msg = None
    assistant_msg = None

    # 1. system
    for msg in messages:
        if msg.get("role") == "system":
            system_msg = msg.get("content")
            break

    # 2. user
    for msg in messages:
        if msg.get("role") == "user":
            user_msg = msg.get("content")
            break

    # 3. assistant (final clean)
    for msg in reversed(messages):
        if msg.get("role") == "assistant":

            # skip tool calls
            if msg.get("tool_calls"):
                continue

            content = msg.get("content")
            content = clean_content(content)

            # skip empty
            if not content or content.strip() == "":
                continue

            # skip reasoning chatter
            if content.lower().startswith("i'll") or content.lower().startswith("i will"):
                continue

            assistant_msg = content
            break

    # validation
    if not system_msg:
        return None, "missing_system"

    if not user_msg:
        return None, "missing_user"

    if not assistant_msg:
        return None, "missing_assistant"

    # optional: filter short outputs (important for Medusa)
    if len(assistant_msg.split()) < 30:
        return None, "too_short"

    return {
        "messages": [
            {"role": "system", "content": system_msg.strip()},
            {"role": "user", "content": user_msg.strip()},
            {"role": "assistant", "content": assistant_msg.strip()}
        ]
    }, "valid"


def process_samples(samples: List[Dict[str, Any]], stats: Dict[str, int]):
    """Process samples and extract clean tool-calling conversations."""
    for sample in samples:
        messages = sample.get("messages", [])
        
        if not messages:
            stats["invalid_format"] += 1
            continue
        
        # Build conversation for tool-calling format
        # Keep the full conversation including tool_calls
        conversation = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            
            # Skip empty messages
            if not content and not tool_calls:
                continue
            
            msg_obj = {"role": role}
            
            if content:
                msg_obj["content"] = str(content) if content else ""
            
            # Add tool_calls if present (this is what we want!)
            if tool_calls:
                msg_obj["tool_calls"] = tool_calls
            
            conversation.append(msg_obj)
        
        # Validate we have at least user + assistant (system is optional for HF)
        roles = [m.get("role") for m in conversation]
        
        # Allow both formats: system+user+assistant OR user+assistant
        has_user = "user" in roles
        has_assistant = "assistant" in roles
        
        if not has_user:
            stats["missing_user"] += 1
            continue
        if not has_assistant:
            stats["missing_assistant"] += 1
            continue
        
        # If we have system, keep it. If not, add a generic one for consistency
        if "system" not in roles:
            # For HF samples without system, add a default one
            source = sample.get("_source", "unknown")
            if source == "huggingface":
                # Insert default system message for HF samples
                conversation.insert(0, {
                    "role": "system",
                    "content": "You are a helpful assistant."
                })
        
        # Check that there's at least one tool_call
        has_tool = any(m.get("tool_calls") for m in conversation)
        if not has_tool:
            stats["no_tool_calls"] += 1
            continue
        
        yield {
            "messages": conversation
        }


def main():
    parser = argparse.ArgumentParser(
        description="Convert datasets to OpenAI format for evaluation"
    )
    parser.add_argument(
        "--source", 
        choices=["local", "hf", "both"], 
        default="local",
        help="Data source: local files, HuggingFace, or both"
    )
    parser.add_argument(
        "--input-dir", 
        default=DEFAULT_INPUT_DIR,
        help="Input directory for local JSON files"
    )
    parser.add_argument(
        "--output", 
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSONL file"
    )
    parser.add_argument(
        "--hf-limit",
        type=int,
        default=None,
        help="Limit number of samples from HuggingFace (for testing)"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=0,
        help="Minimum words in assistant response (0 = no filter)"
    )
    
    args = parser.parse_args()
    
    # Collect all samples
    all_samples = []
    
    if args.source in ["local", "both"]:
        local_samples = load_from_local(args.input_dir)
        all_samples.extend(local_samples)
    
    if args.source in ["hf", "both"]:
        if not HF_AVAILABLE:
            print("Error: HuggingFace datasets not available. Install: pip install datasets")
            return
        hf_samples = load_from_huggingface(limit=args.hf_limit)
        all_samples.extend(hf_samples)
    
    if not all_samples:
        print("No samples found!")
        return
    
    print(f"\nTotal samples to process: {len(all_samples)}")
    
    # Stats
    stats = {
        "missing_system": 0,
        "missing_user": 0,
        "missing_assistant": 0,
        "invalid_format": 0,
        "too_short": 0,
        "no_tool_calls": 0
    }
    
    count_valid = 0
    
    with open(args.output, "w") as out_f:
        for sample in process_samples(all_samples, stats):
            # Apply minimum word filter if specified
            if args.min_words > 0:
                # Find assistant message
                assistant_content = None
                for msg in sample.get("messages", []):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        assistant_content = msg.get("content", "")
                        break
                
                if assistant_content and len(assistant_content.split()) < args.min_words:
                    stats["too_short"] += 1
                    continue
            
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count_valid += 1

    print("\n===== SUMMARY =====")
    print(f"Total samples: {len(all_samples)}")
    print(f"Valid samples: {count_valid}")
    print(f"Dropped samples: {len(all_samples) - count_valid}")

    print("\nDrop reasons:")
    for k, v in stats.items():
        if v > 0:
            print(f"  {k}: {v}")

    print(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()
