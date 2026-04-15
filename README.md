# Speculative-Decoding-Parser 🔐

A multilingual evaluation dataset generator for payment transaction analysis. This project processes conversation data with tool_calls from local processed files, HuggingFace datasets, or GCS buckets into an OpenAI-compatible or ShareGPT format.

## Project Overview 📋

This dataset is designed for training and evaluating AI models on payment transaction log analysis tasks. It contains conversations with:

- 📝 System prompts defining the role of a payment transaction analysis agent
- ❓ User queries about order investigations, payment failures, and transaction debugging
- 🤖 Assistant responses with tool_calls to various analysis tools

### Data Sources 📦

1. **Local processed files** (`processed/`): JSON files from SSH machine data containing payment transaction logs
2. **HuggingFace** (`Salesforce/xlam-function-calling-60k`): Function calling dataset for additional training data
3. **GCS Bucket**: Google Cloud Storage bucket for cloud-based data sources

## Project Structure 📂

```
juspay-eval-multilingual/
├── output/            # Converted output files (JSONL/JSON)
│   └── dataset_YYYYMMDD_HHMMSS.jsonl
├── processed/        # Local JSON files with tool_calls (from SSH machine)
├── raw/              # Raw chat completion data
├── generate_data.py  # Main data generation script
├── requirements.txt  # Python dependencies
├── .env              # HuggingFace token and GCP credentials (sensitive)
├── .env.example      # Template for .env
└── example.json      # Example data format
```

## Installation 🛠️

```bash
# Install dependencies
pip install -r requirements.txt
```

Note: On macOS, you can also install via brew:
```bash
brew install gitleaks
```

## Configuration ⚙️

1. Copy the example env file:
```bash
cp .env.example .env
```

2. Add your HuggingFace token to `.env`:
```
HF_TOKEN=your_huggingface_token_here
```

3. (Optional) Add GCP credentials for GCS bucket scanning:
```
GCP_PROJECT_ID=your_project_id
# OR
GCP_SERVICE_ACCOUNT_KEY=/path/to/service-account-key.json
# OR
GCP_CREDENTIALS_B64=your_base64_encoded_credentials
```

🔑 Get a HuggingFace token from: https://huggingface.co/settings/tokens

## Usage 🚀

### Basic Usage

```bash
# Generate dataset from local files only (10 samples, OpenAI format)
python3 generate_data.py -n 10 -o ./output -f openai --local

# Generate dataset from local files (ShareGPT format)
python3 generate_data.py -n 10 -o ./output -f sharegpt --local

# Generate dataset from HuggingFace only
python3 generate_data.py -n 10 -o ./output -f openai --hf

# Generate dataset from GCS bucket
python3 generate_data.py -n 10 -o ./output -f openai --gcs my-bucket
```

### Multiple Sources with Ratio

```bash
# Mix HuggingFace and local sources with ratio
python3 generate_data.py -n 100 -o ./output -f openai --hf --local --ratio hf:0.6,local:0.4
```

### Additional Options

```bash
# With minimum word filter for assistant responses
python3 generate_data.py -n 10 -o ./output -f openai --local --min-words 30

# Stop processing if secrets are detected
python3 generate_data.py -n 10 -o ./output -f openai --local --stop-on-secret

# Remove duplicate samples
python3 generate_data.py -n 10 -o ./output -f openai --local --deduplicate

# Output as JSON instead of JSONL
python3 generate_data.py -n 10 -o ./output -f openai --local --output-format json

# Resume from existing output file
python3 generate_data.py -n 10 -o ./output -f openai --local --resume ./output/existing.jsonl

# Custom input directory
python3 generate_data.py -n 10 -o ./output --local --input-dir ./my-data

# Custom HuggingFace dataset
python3 generate_data.py -n 10 -o ./output --hf --hf-dataset another/dataset
```

#### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-o`, `--output` | Output directory | `./output` |
| `-f`, `--format` | Output format: `openai` or `sharegpt` | `openai` |
| `-n`, `--num-samples` | Total number of samples to generate | `1000` |
| `--hf` | Use HuggingFace dataset as source | - |
| `--local` | Use local files as source | - |
| `--gcs` | Use GCS bucket as source | - |
| `--ratio` | Ratio for mixing sources (e.g., `hf:0.6,local:0.4`) | - |
| `--input-dir` | Input directory for local JSON files | `./processed` |
| `--hf-dataset` | HuggingFace dataset name | `Salesforce/xlam-function-calling-60k` |
| `--min-words` | Minimum words in assistant response (0=disabled) | `0` |
| `--stop-on-secret` | Stop if secrets detected | `false` |
| `--deduplicate` | Remove duplicate samples based on content hash | `false` |
| `--resume` | Resume from existing JSONL/JSON file | - |
| `--output-format` | Output format: `jsonl` or `json` | `jsonl` |

## Output Format 📄

### OpenAI Format (default)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the status of order 12345?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "get_order_status",
          "arguments": "{\"order_id\": \"12345\"}"
        }
      }
    ]},
    {"role": "tool", "content": "Order status: CONFIRMED", "tool_call_id": "call_123"},
    {"role": "assistant", "content": "The order 12345 is CONFIRMED."}
  ]
}
```

### ShareGPT Format

```json
{
  "conversations": [
    {"from": "human", "value": "You are a helpful assistant."},
    {"from": "human", "value": "What is the status of order 12345?"},
    {"from": "gpt", "value": "\n[Tool Calls]:\n- get_order_status({\"order_id\": \"12345\"})\n\nOrder status: CONFIRMED"}
  ]
}
```

> ⚠️ Note: ShareGPT format converts tool_calls to text representation since ShareGPT doesn't natively support them.

## System Resource Monitoring 💻

The script displays system resources before generating the dataset:

```
============================================================
SYSTEM RESOURCES
============================================================

CPU:
  Cores: 14 logical, 14 physical
  Usage: 18.2%

Memory:
  Total: 24.0 GB
  Available: 5.25 GB
  Usage: 78.1%

Storage (current disk):
  Total: 460.43 GB
  Free: 130.0 GB
  Usage: 71.8%

Estimated output size: ~2 KB per sample
```

## Sensitive Data Scanning 🔒

The script includes a **sensitive data scanner** to detect API keys, tokens, passwords, and other secrets using gitleaks, trufflehog, or detect-secrets.

### Scan local directory

```bash
python3 generate_data.py scan-secrets --path .
```

### Scan specific directory

```bash
python3 generate_data.py scan-secrets --path ./processed
```

### Scan GCS bucket

```bash
python3 generate_data.py scan-secrets --path my-bucket-name
```

### Save findings to JSON

```bash
python3 generate_data.py scan-secrets --path . --output findings.json
```

#### Scan-secrets CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--path` | Path to scan (directory or GCS bucket name) | `.` |
| `--extensions` | File extensions to scan | `.json, .yaml, .yml, .env, .txt, .py, .js, .ts, .sh, .tf, .cfg, .ini, .properties` |
| `--output` | Save findings to JSON file | - |

### Detected Secret Types

- 🔴 **HIGH**: AWS keys, GCP keys, OpenAI API keys, HuggingFace tokens, GitHub/GitLab tokens, Stripe keys, passwords, database URLs with credentials, private keys, PayPal/Razorpay keys
- 🟡 **MEDIUM**: Generic API keys, Stripe test keys, environment variable references

> ✅ Secret scanning is **always enabled** during data generation. Use `--stop-on-secret` to halt processing when secrets are detected.

## Dataset Statistics 📊

- 📁 Local processed files: ~96 samples with tool_calls
- 🤗 HuggingFace (xlam-function-calling-60k): 60,000 samples
- ➕ Combined output: Validated samples with system+user+assistant+tool_calls

## License 📜

See LICENSE file for details.
