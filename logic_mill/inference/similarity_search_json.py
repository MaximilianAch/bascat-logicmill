import json
import os
import sys
import argparse
from urllib3.util import Retry
from pathlib import Path
from datetime import datetime
from requests import Session
from requests.adapters import HTTPAdapter

API_URL = 'https://api.logic-mill.net/api/v1/graphql/'
DEFAULT_MODEL = "patspecter"
DEFAULT_AMOUNT = 25
DEFAULT_SEARCH_TYPE = 'both'
RETRY_TOTAL = 5
RETRY_BACKOFF = 0.1
RETRY_STATUS_CODES = [500, 501, 502, 503, 504, 524]

GRAPHQL_QUERY = """
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


def parse_args():
    parser = argparse.ArgumentParser(description='Search for similar patents and publications using JSON input')
    parser.add_argument('input_file', help='JSON input file with title and abstract')
    parser.add_argument('--amount', '-n', type=int, help=f'Number of results (overrides JSON, default: {DEFAULT_AMOUNT})')
    parser.add_argument('--type', choices=['patents', 'publications', 'both'], help=f'Search in patents, publications, or both (overrides JSON, default: {DEFAULT_SEARCH_TYPE})')
    parser.add_argument('--output', '-o', help='Output file (default: auto-generated in results/)')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    return parser.parse_args()


def load_input(input_file):
    try:
        with open(input_file) as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        exit(1)

    if 'title' not in input_data or 'abstract' not in input_data:
        print("Error: Input JSON must contain 'title' and 'abstract' fields", file=sys.stderr)
        exit(1)

    return input_data


def create_session():
    token = os.getenv('LOGICMILL_API_TOKEN')
    if not token:
        print("Error: LOGICMILL_API_TOKEN environment variable not set")
        print("Please set it with: export LOGICMILL_API_TOKEN='your_token_here'")
        exit(1)

    session = Session()
    retries = Retry(total=RETRY_TOTAL, backoff_factor=RETRY_BACKOFF,
                    status_forcelist=RETRY_STATUS_CODES)
    session.mount('https://', HTTPAdapter(max_retries=retries))

    return session, token


def fetch_results(session, token, input_data, amount, search_type):
    if search_type == 'patents':
        indices = ["patents"]
    elif search_type == 'publications':
        indices = ["publications"]
    else:
        indices = ["patents", "publications"]

    variables = {
        "model": DEFAULT_MODEL,
        "data": [
            {
                "key": "title",
                "value": input_data['title']
            },
            {
                "key": "abstract",
                "value": input_data['abstract']
            }
        ],
        "amount": amount,
        "indices": indices
    }

    headers = {
        'content-type': 'application/json',
        'Authorization': f'Bearer {token}',
    }

    r = session.post(API_URL, headers=headers, json={'query': GRAPHQL_QUERY, 'variables': variables})

    if r.status_code != 200:
        print(f"Error: Request failed with status code {r.status_code}", file=sys.stderr)
        print(r.text, file=sys.stderr)
        exit(1)

    response = r.json()

    if 'errors' in response:
        print("API Error:", file=sys.stderr)
        print(json.dumps(response['errors'], indent=2), file=sys.stderr)
        exit(1)

    return response.get('data', {}).get('encodeDocumentAndSimilaritySearch', [])

def save_results(results, input_file, output_arg, json_format):
    results_dir = Path(__file__).parent / "results"
    results_json_dir = results_dir / "json"
    results_dir.mkdir(exist_ok=True)
    results_json_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(input_file).stem

    json_archive_path = results_json_dir / f"{base_name}_{timestamp}.json"
    with open(json_archive_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON archive to: {json_archive_path}")

    if output_arg:
        output_path = Path(output_arg)
    else:
        ext = ".json" if json_format else ".txt"
        output_path = results_dir / f"{base_name}_{timestamp}{ext}"

    with open(output_path, 'w') as output_file:
        if json_format:
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

    format_type = "JSON" if json_format else "text"
    print(f"Saved {format_type} results to: {output_path}")


def display_results(results):
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
        if doc.get('PatspecterEmbedding'):
            embedding = doc['PatspecterEmbedding']
            print(f"   Embedding: [{embedding[0]:.4f}, {embedding[1]:.4f}, ... {embedding[-1]:.4f}] (dim: {len(embedding)})")
        print()

    total_patents = sum(1 for r in results if r.get('index') == 'patents')
    total_publications = sum(1 for r in results if r.get('index') == 'publications')
    print("=" * 60)
    print(f"Summary: {total_patents} patents, {total_publications} publications")


def main():
    args = parse_args()
    input_data = load_input(args.input_file)
    session, token = create_session()

    amount = args.amount if args.amount is not None else input_data.get('amount', DEFAULT_AMOUNT)
    search_type = args.type if args.type else input_data.get('type', DEFAULT_SEARCH_TYPE)

    results = fetch_results(session, token, input_data, amount, search_type)
    save_results(results, args.input_file, args.output, args.json)
    display_results(results)


if __name__ == "__main__":
    main()
