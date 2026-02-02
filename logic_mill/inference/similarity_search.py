import json
import os
from urllib3.util import Retry
from requests import Session
from requests.adapters import HTTPAdapter

TOKEN = os.getenv('LOGICMILL_API_TOKEN')
if not TOKEN:
    print("Error: LOGICMILL_API_TOKEN environment variable not set")
    print("Please set it with: export LOGICMILL_API_TOKEN='your_token_here'")
    exit(1)

s = Session()
retries = Retry(total=5, backoff_factor=0.1,
                status_forcelist=[500, 501, 502, 503, 504, 524])
s.mount('https://', HTTPAdapter(max_retries=retries))

URL = 'https://api.logic-mill.net/api/v1/graphql/'
headers = {
  'content-type': 'application/json',
  'Authorization': 'Bearer '+ TOKEN,
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

print("Patent/Publication Similarity Search")
print("=" * 60)

title = input("Enter title: ").strip()
if not title:
    title = "Untitled"

print("\nEnter abstract/summary (press Ctrl+D when finished):")
abstract_lines = []
try:
    while True:
        line = input()
        abstract_lines.append(line)
except EOFError:
    pass

abstract = "\n".join(abstract_lines).strip()

if not abstract:
    print("No abstract provided. Exiting.")
    exit(1)

amount = input("\nNumber of results to return (default: 25): ").strip()
amount = int(amount) if amount.isdigit() else 25

search_type = input("\nSearch in (patents/publications/both, default: both): ").strip().lower()
if search_type == "patents":
    indices = ["patents"]
elif search_type == "publications":
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

print("\n" + "=" * 60)
print("Searching for similar patents and publications...")
print("=" * 60)

r = s.post(URL, headers=headers, json={'query': query, 'variables': variables})

if r.status_code != 200:
    print(f"Error: Request failed with status code {r.status_code}")
    print(r.text)
    exit(1)

response = r.json()

if 'errors' in response:
    print("API Error:")
    print(json.dumps(response['errors'], indent=2))
    exit(1)

results = response.get('data', {}).get('encodeDocumentAndSimilaritySearch', [])

if not results:
    print("No results found.")
    exit(0)

print(f"\nFound {len(results)} similar documents:\n")

patent_count = 0
publication_count = 0

for i, result in enumerate(results, 1):
    doc = result.get('document', {})
    index_type = result.get('index', 'unknown')

    if index_type == 'patents':
        patent_count += 1
    elif index_type == 'publications':
        publication_count += 1

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

print("=" * 60)
print(f"Summary: {patent_count} patents, {publication_count} publications")
