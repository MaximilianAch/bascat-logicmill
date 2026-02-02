import requests
import json
import sys
from pathlib import Path

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
elif host is None:
    host = "daisg106"

if len(sys.argv) > 2:
    port = int(sys.argv[2])
elif port is None:
    port = 9732

url = f"http://{host}:{port}/v1/chat/completions"

print(f"Connected to {host}:{port}")
print("Interactive Chat Mode - Type 'exit' or 'quit' to end the conversation")
print("=" * 60)

messages = []

while True:
    try:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": "/u/achmaxim/Research/llms/local_llms/data/OSS-120G",
            "messages": messages,
            "max_tokens": 20000,
            "temperature": 0.0
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        assistant_message = result["choices"][0]["message"]["content"]

        messages.append({"role": "assistant", "content": assistant_message})

        print(f"\nAssistant: {assistant_message}")
        print(f"\n[Tokens: {result['usage']['total_tokens']}]")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except requests.exceptions.RequestException as e:
        print(f"\nError connecting to server: {e}")
        break
    except Exception as e:
        print(f"\nError: {e}")
        continue
