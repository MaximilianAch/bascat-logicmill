import requests
import json
import sys
import os
import argparse
import textwrap
from pathlib import Path
from urllib3.util import Retry
from requests import Session
from requests.adapters import HTTPAdapter

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

    if 'errors' in response:
        print("\nAPI Error:", file=sys.stderr)
        print(json.dumps(response['errors'], indent=2), file=sys.stderr)
        sys.exit(1)

    return response.get('data', {}).get('encodeDocumentAndSimilaritySearch', [])

def main():
    parser = argparse.ArgumentParser(
        description='Generate summary from inputs and search for similar patents/publications'
    )
    parser.add_argument('--host', help='vLLM server hostname (overrides config file)')
    parser.add_argument('--port', type=int, help='vLLM server port (overrides config file)')
    parser.add_argument('--amount', '-n', type=int, default=25, help='Number of results (default: 25)')
    parser.add_argument('--type', choices=['patents', 'publications', 'both'], default='both',
                        help='Search in patents, publications, or both (default: both)')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--json', action='store_true', help='Output search results in JSON format')
    parser.add_argument('--save-summary', help='Save generated title and abstract to JSON file')

    args = parser.parse_args()

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
    print("Inputs to Patent/Publication Search")
    print("=" * 60)
    print("Enter your inputs (one per line)")
    print("Type 'END' on a new line when finished, or Ctrl+D")
    print("=" * 60)

    inputs = []

    try:
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            if line.strip():
                inputs.append(line.strip())
    except EOFError:
        pass

    if not inputs:
        print("\nNo inputs provided. Exiting.")
        sys.exit(0)

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

    if args.save_summary:
        with open(args.save_summary, 'w') as f:
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
        print(f"\nSaved summary to: {args.save_summary}")

    print("\n" + "=" * 60)
    print("Searching for similar patents/publications...")
    print("=" * 60)

    results = search_similar_patents(title, abstract, args.amount, args.type)

    if args.output:
        with open(args.output, 'w') as output_file:
            if args.json:
                json.dump(results, output_file, indent=2)
            else:
                print(f"\nFound {len(results)} similar documents:\n", file=output_file)

                patent_count = 0
                publication_count = 0

                for i, result in enumerate(results, 1):
                    doc = result.get('document', {})
                    index_type = result.get('index', 'unknown')

                    if index_type == 'patents':
                        patent_count += 1
                    elif index_type == 'publications':
                        publication_count += 1

                    print(f"{i}. {doc.get('title', 'No title')}", file=output_file)
                    print(f"   Type: {index_type}", file=output_file)
                    print(f"   Score: {result.get('score', 0):.4f}", file=output_file)
                    print(f"   ID: {result.get('id', 'N/A')}", file=output_file)
                    if doc.get('url'):
                        print(f"   URL: {doc['url']}", file=output_file)
                    print(file=output_file)

                print("=" * 60, file=output_file)
                print(f"Summary: {patent_count} patents, {publication_count} publications", file=output_file)

        print(f"\nResults saved to: {args.output}")

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
