import requests
import json
import sys
import os
import argparse
import textwrap
from pathlib import Path
from datetime import datetime
from urllib3.util import Retry
from requests import Session
from requests.adapters import HTTPAdapter

DEFAULT_INPUTS = [
    "Novel method for improving neural network efficiency",
    "Uses sparse attention mechanisms to reduce computational cost",
    "Achieves 40% speedup with minimal accuracy loss",
    "Applicable to transformer architectures",
    "Tested on multiple NLP benchmarks"
]

def load_vllm_config():
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
                print(f"Loaded vLLM server info from {config_file}")
                print(f"  Server started at: {config.get('started_at', 'unknown')}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read config file: {e}")

    return host, port

def generate_summary_and_title(inputs, host, port):
    url = f"http://{host}:{port}/v1/chat/completions"

    inputs_text = "\n".join(f"{point} " for point in inputs)

    prompt = f"""I want to search for relevant prior art—especially patents—using a sentence transformer. 
    Based on the scientific text provided below, generate:

    1. A concise, descriptive scientific title (one line) that explicitly includes key words from the provided text.

    2. A patent-style abstract (1–2 paragraphs) that is technically accurate, contains all key 
    details present in the source text, and is written with clarity suitable for similarity 
    search against patents and scientific publications. 
    - Do not add or infer any information that is not given.

    Input scientific text:
    {inputs_text}

    Format the output exactly as follows:

    TITLE: [your title here]
    ABSTRACT: [your abstract here]
    """

    payload = {
        "model": "/u/achmaxim/Research/llms/local_llms/data/OSS-120G",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "typical_p": 1.0,
        "do_sample": False,
        "seed": 42
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

        print(f"\nTokens used: {result['usage']['total_tokens']}")

        return title, abstract

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to vLLM server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating summary: {e}")
        sys.exit(1)

def search_similar_patents(title, abstract, amount, search_type):
    TOKEN = os.getenv('LOGICMILL_API_TOKEN')
    if not TOKEN:
        print("\nError: LOGICMILL_API_TOKEN environment variable not set")
        print("Please set it with: export LOGICMILL_API_TOKEN='your_token_here'")
        sys.exit(1)

    s = Session()
    retries = Retry(total=5, backoff_factor=0.1,
                    status_forcelist=[500, 501, 502, 503, 504, 524])
    s.mount('https://', HTTPAdapter(max_retries=retries))

    URL = 'https://api.logic-mill.net/api/v1/graphql/'
    headers = {
        'content-type': 'application/json',
        'Authorization': 'Bearer ' + TOKEN,
    }

    query = """
query embedDocumentAndSimilaritySearch($data: [EncodeDocumentPart], $indices: [String], $amount: Int, $model: String!) {
  encodeDocumentAndSimilaritySearch(
    data: $data
    indices: $indices
    amount: $amount
    model: $model
  ) {
    id
    score
    index
    document {
      title
      url
      PatspecterEmbedding
    }
  }
}
"""

    if search_type == 'patents':
        indices = ["patents"]
    elif search_type == 'publications':
        indices = ["publications"]
    else:
        indices = ["patents", "publications"]

    variables = {
        "model": "patspecter",
        "data": [
            {
                "key": "title",
                "value": title
            },
            {
                "key": "abstract",
                "value": abstract
            }
        ],
        "amount": amount,
        "indices": indices
    }

    r = s.post(URL, headers=headers, json={'query': query, 'variables': variables})

    if r.status_code != 200:
        print(f"\nError: Request failed with status code {r.status_code}", file=sys.stderr)
        print(r.text, file=sys.stderr)
        sys.exit(1)

    response = r.json()
    # print("RESPONSE KEYS\n", response['data']['encodeDocumentAndSimilaritySearch'][0].keys())
    # print("RESPONSE DOCUMENT KEYS\n", response['data']['encodeDocumentAndSimilaritySearch'][0]['document'].keys())
    if 'errors' in response:
        print("\nAPI Error:", file=sys.stderr)
        print(json.dumps(response['errors'], indent=2), file=sys.stderr)
        sys.exit(1)

    return response.get('data', {}).get('encodeDocumentAndSimilaritySearch', [])

def main():
    parser = argparse.ArgumentParser(
        description='Generate an abstract from inputs and search for similar patents/publications'
    )
    parser.add_argument('input', help='JSON file with inputs (required)')
    parser.add_argument('--host', help='vLLM server hostname (overrides config file)')
    parser.add_argument('--port', type=int, help='vLLM server port (overrides config file)')
    parser.add_argument('--amount', '-n', type=int, default=25, help='Number of results (default: 25)')
    parser.add_argument('--type', choices=['patents', 'publications', 'both'], default='both',
                        help='Search in patents, publications, or both (default: both)')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--json', action='store_true', help='Output search results in JSON format')

    args = parser.parse_args()

    inputs = DEFAULT_INPUTS
    source = "hardcoded"
    base_name = "default"

    if args.input:
        base_name = Path(args.input).stem
        try:
            with open(args.input) as f:
                input_data = json.load(f)
            if 'inputs' in input_data:
                inputs = input_data['inputs']
                source = f"file: {args.input}"
            else:
                print(f"Warning: JSON file '{args.input}' does not contain 'inputs' field")
                print("Using hardcoded inputs instead")
        except FileNotFoundError:
            print(f"Error: Input file '{args.input}' not found")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in input file: {e}")
            sys.exit(1)

    host = args.host
    port = args.port

    if not host or not port:
        config_host, config_port = load_vllm_config()
        host = host or config_host
        port = port or config_port

    if not host or not port:
        print("Error: Could not determine vLLM server host/port")
        print("Please provide --host and --port or ensure config file exists")
        sys.exit(1)

    print(f"\nConnected to vLLM server: {host}:{port}")
    print("=" * 60)
    print("Inputs to Similarity Search")
    print("=" * 60)
    print(f"Using inputs from: {source}")
    for i, point in enumerate(inputs, 1):
        print(f"  {i}. {point}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Generating summary and title...")
    print("=" * 60)

    title, abstract = generate_summary_and_title(inputs, host, port)

    print("\nGENERATED TITLE:")
    print("-" * 60)
    print(title)
    print("\nGENERATED ABSTRACT:")
    print("-" * 60)
    print(abstract)
    print("-" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries_dir = Path(__file__).parent / "summaries"
    summaries_dir.mkdir(exist_ok=True)

    summary_file = summaries_dir / f"{base_name}_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'title': title,
            'abstract': abstract,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary to: {summary_file}")

    print("\n" + "=" * 60)
    print("Searching for similar patents/publications...")
    print("=" * 60)

    results = search_similar_patents(title, abstract, args.amount, args.type)

    results_dir = Path(__file__).parent / "results"
    results_json_dir = Path(__file__).parent / "results/json"
    results_dir.mkdir(exist_ok=True)
    results_json_dir.mkdir(exist_ok=True)

    default_output = results_dir / f"{base_name}_{timestamp}.txt"
    json_output = results_json_dir / f"{base_name}_{timestamp}.json"

    output_path = args.output if args.output else default_output

    with open(json_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON results to: {json_output}")

    with open(output_path, 'w') as output_file:
        if args.json:
            json.dump(results, output_file, indent=2)
        else:
            patents = [r for r in results if r.get('index') == 'patents']
            publications = [r for r in results if r.get('index') == 'publications']

            print(f"\nFound {len(results)} similar documents ({len(patents)} patents, {len(publications)} publications):\n", file=output_file)

            if patents:
                print("=" * 60, file=output_file)
                print("PATENTS", file=output_file)
                print("=" * 60, file=output_file)
                print(file=output_file)

                for i, result in enumerate(patents, 1):
                    doc = result.get('document', {})
                    print(f"{i}. {doc.get('title', 'No title')}", file=output_file)
                    print(f"   Score: {result.get('score', 0):.4f}", file=output_file)
                    print(f"   ID: {result.get('id', 'N/A')}", file=output_file)
                    if doc.get('url'):
                        print(f"   URL: {doc['url']}", file=output_file)
                    print(file=output_file)

            if publications:
                print("=" * 60, file=output_file)
                print("PUBLICATIONS", file=output_file)
                print("=" * 60, file=output_file)
                print(file=output_file)

                for i, result in enumerate(publications, 1):
                    doc = result.get('document', {})
                    print(f"{i}. {doc.get('title', 'No title')}", file=output_file)
                    print(f"   Score: {result.get('score', 0):.4f}", file=output_file)
                    print(f"   ID: {result.get('id', 'N/A')}", file=output_file)
                    if doc.get('url'):
                        print(f"   URL: {doc['url']}", file=output_file)
                    print(file=output_file)

            print("=" * 60, file=output_file)
            print(f"Summary: {len(patents)} patents, {len(publications)} publications", file=output_file)

    print(f"\nSaved text results to: {output_path}")

    if len(results) > 3:
        print(f"\nFound {len(results)} similar documents (showing first 3 on terminal):\n")
    else:
        print(f"\nFound {len(results)} similar documents:\n")

    display_results = results[:3]
    for i, result in enumerate(display_results, 1):
        doc = result.get('document', {})
        index_type = result.get('index', 'unknown')

        print(f"{i}. {doc.get('title', 'No title')}")
        print(f"   Type: {index_type}")
        print(f"   Score: {result.get('score', 0):.4f}")
        print(f"   ID: {result.get('id', 'N/A')}")
        if doc.get('url'):
            print(f"   URL: {doc['url']}")
        print()

    total_patents = sum(1 for r in results if r.get('index') == 'patents')
    total_publications = sum(1 for r in results if r.get('index') == 'publications')
    print("=" * 60)
    print(f"Summary: {total_patents} patents, {total_publications} publications")

if __name__ == "__main__":
    main()
