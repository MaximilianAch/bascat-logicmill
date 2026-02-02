import requests
import json
import sys
import textwrap
from pathlib import Path

INPUTS = [
    "Novel method for improving neural network efficiency",
    "Uses sparse attention mechanisms to reduce computational cost",
    "Achieves 40% speedup with minimal accuracy loss",
    "Applicable to transformer architectures",
    "Tested on multiple NLP benchmarks"
]

base_dir = Path(__file__).parent.parent
config_8gpu = base_dir / "vllm_server_info_8gpu.json"
config_4gpu = base_dir / "vllm_server_info_4gpu.json"
config_2gpu = base_dir / "vllm_server_info_2gpu.json"
config_1gpu = base_dir / "vllm_server_info_1gpu.json"

config_file = None
for cfg in [config_8gpu, config_4gpu, config_2gpu, config_1gpu]:
    if cfg.exists():
        config_file = cfg
        break

host = None
port = None

if config_file:
    try:
        with open(config_file) as f:
            config = json.load(f)
            host = config.get("hostname")
            port = config.get("port")
            print(f"Loaded server info from {config_file}")
            print(f"  Server started at: {config.get('started_at', 'unknown')}")
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read config file: {e}")

if len(sys.argv) > 1:
    host = sys.argv[1]

if len(sys.argv) > 2:
    port = int(sys.argv[2])

if not host or not port:
    print("Error: Could not determine vLLM server host/port")
    print("Please provide host and port as arguments or ensure config file exists")
    sys.exit(1)

url = f"http://{host}:{port}/v1/chat/completions"

print(f"Connected to {host}:{port}")
print("Inputs to Summary/Abstract Generator (Hardcoded)")
print("=" * 60)
print("Using hardcoded inputs:")
for i, point in enumerate(INPUTS, 1):
    print(f"  {i}. {point}")
print("=" * 60)

inputs = INPUTS

print("\n" + "=" * 60)
print("Generating summary...")
print("=" * 60)

inputs_text = "\n".join(f"• {point}" for point in inputs)

prompt = f"""Based on the following inputs, generate:
1. A descriptive title (one line)
2. A well-written summary or abstract (1-2 paragraphs)

Inputs:
{inputs_text}

Please format your response as:
TITLE: [your title here]
ABSTRACT: [your abstract here]

Do not make up any information."""

payload = {
    "model": "/u/achmaxim/Research/llms/local_llms/data/OSS-120G",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 1000,
    "temperature": 0.0
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    title = ""
    abstract = ""

    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("TITLE:"):
            title = line.replace("TITLE:", "").strip()
        elif line.startswith("ABSTRACT:"):
            abstract = "\n".join(lines[i:]).replace("ABSTRACT:", "").strip()
            break

    if not title or not abstract:
        title = "Generated Summary"
        abstract = content

    print("\nGENERATED TITLE:")
    print("-" * 60)
    print(title)
    print("\nGENERATED ABSTRACT:")
    print("-" * 60)
    print(abstract)
    print("-" * 60)
    print(f"\nTokens used: {result['usage']['total_tokens']}")

    summaries_dir = Path(__file__).parent / "summaries"
    summaries_dir.mkdir(exist_ok=True)

    output_file = summaries_dir / "generated_summary.json"
    with open(output_file, 'w') as f:
        abstract_lines = []
        for para in abstract.split('\n'):
            if para.strip():
                abstract_lines.extend(textwrap.wrap(para, width=80))
                abstract_lines.append("")
        if abstract_lines and abstract_lines[-1] == "":
            abstract_lines.pop()

        json.dump({
            'title': title,
            'abstract': abstract,
            'abstract_lines': abstract_lines
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_file}")
    print(f"\nYou can now run:")
    print(f"  python inference/similarity_search_json.py {output_file}")

except requests.exceptions.RequestException as e:
    print(f"Error connecting to server: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
