# Speculative-Decoding-Parser

A multilingual evaluation dataset for payment transaction analysis. This project processes conversation data with tool_calls from local processed files and HuggingFace datasets into an OpenAI-compatible format.

## Project Overview

This dataset is designed for training and evaluating AI models on payment transaction log analysis tasks. It contains conversations with:

- System prompts defining the role of a payment transaction analysis agent
- User queries about order investigations, payment failures, and transaction debugging
- Assistant responses with tool_calls to various analysis tools

### Data Sources

1. **Local processed files** (`processed/`): JSON files from SSH machine data containing payment transaction logs
2. **HuggingFace** (`Salesforce/xlam-function-calling-60k`): Function calling dataset for additional training data

## Project Structure

```
juspay-eval-multilingual/
├── output/            # Converted output files (JSONL)
│   ├── openai_dataset.jsonl
│   └── hf_output.jsonl
├── processed/        # Local JSON files with tool_calls (from SSH machine)
├── raw/              # Raw chat completion data
├── openai_format.py  # Main converter script
├── requirements.txt  # Python dependencies
├── .env              # HuggingFace token (sensitive)
├── .env.example      # Template for .env
└── example.json      # Example data format
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy the example env file:
```bash
cp .env.example .env
```

2. Add your HuggingFace token to `.env`:
```
HF_TOKEN=your_huggingface_token_here
```

Get a token from: https://huggingface.co/settings/tokens

## Usage

### Convert local files only
```bash
python3 openai_format.py --source local
```

### Convert HuggingFace dataset only
```bash
python3 openai_format.py --source hf --hf-limit 1000
```

### Combine both sources
```bash
python3 openai_format.py --source both --hf-limit 50
```

### Additional options
```bash
python3 openai_format.py --source both --hf-limit 100 --output custom_output.jsonl --min-words 30
```

#### CLI Arguments
- `--source`: Data source - `local`, `hf`, or `both` (default: local)
- `--input-dir`: Input directory for local JSON files (default: processed/)
- `--output`: Output JSONL file (default: openai_dataset.jsonl)
- `--hf-limit`: Limit number of samples from HuggingFace
- `--min-words`: Minimum words in assistant response (0 = no filter)

## Output Format

The converter produces JSONL files in OpenAI format:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [...]}
  ]
}
```

## Dataset Statistics

- Local processed files: ~96 samples with tool_calls
- HuggingFace (xlam-function-calling-60k): 60,000 samples
- Combined output: Validated samples with system+user+assistant+tool_calls

## Usage - Sensitive Data Scanning

The script also includes a **sensitive data scanner** to detect API keys, tokens, passwords, and other secrets in local files or GCS buckets.

### Scan local directory
```bash
python3 openai_format.py scan-secrets --path .
```

### Scan specific directory
```bash
python3 openai_format.py scan-secrets --path ./processed
```

### Scan GCS bucket
```bash
python3 openai_format.py scan-secrets --path my-bucket-name --gcs
```

### Scan GCS bucket with prefix
```bash
python3 openai_format.py scan-secrets --path my-bucket-name --gcs --prefix logs/
```

### Save findings to JSON
```bash
python3 openai_format.py scan-secrets --path . --output findings.json
```

#### Scan-secrets CLI Arguments
- `--path`: Path to scan (directory or GCS bucket name)
- `--gcs`: Treat path as GCS bucket name
- `--prefix`: GCS prefix to scan
- `--extensions`: File extensions to scan (default: .json, .yaml, .yml, .env, .txt, .py, .js, .ts, .sh, .tf, .cfg, .ini, .properties)
- `--output`: Save findings to JSON file

### Detected Secret Types
- **HIGH**: AWS keys, GCP keys, OpenAI API keys, HuggingFace tokens, GitHub/GitLab tokens, Stripe keys, passwords, database URLs with credentials, private keys, PayPal/Razorpay keys
- **MEDIUM**: Generic API keys, Stripe test keys, environment variable references

## License

See LICENSE file for details.
