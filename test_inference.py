import os
import json
import textwrap
from openai import OpenAI
from openenv.core.client import EnvClient

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8b-chat-hf")
MAX_STEPS = 8
TEMPERATURE = 0.2

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous DevOps agent managing a cloud infrastructure portal.
    You will receive the current state of 5 servers in JSON format.
    Reply with exactly one action in JSON format.
    
    Available Actions:
    1. {"action_type": "query_logs", "target_server_id": "srv-XXX", "severity_filter": "warning", "max_entries": 10}
    2. {"action_type": "modify_resource", "target_server_id": "srv-XXX", "new_tier": "large"}
    3. {"action_type": "delete_resource", "target_server_id": "srv-XXX", "confirm_deletion": true}
    4. {"action_type": "request_human_confirmation", "reasoning_trace": "Explain why you are deferring..."}
    
    Do not output any markdown formatting or explanations outside of the JSON block.
    """
).strip()

def extract_json_action(text: str) -> dict:
    """Attempt to parse the LLM's text output into a dictionary."""
    try:
        clean_text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from model output: {text}")
        return {"action_type": "query_logs", "target_server_id": "srv-000", "severity_filter": "info", "max_entries": 1}

def main():
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable is not set.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print("Connecting to AegisEnv on http://localhost:8000...")
    env = EnvClient("http://localhost:8000")
    
    try:
        obs = env.reset()
        print("\n=== EPISODE STARTED ===")
        print(f"Message: {obs.metadata.get('message')}")

        for step in range(1, MAX_STEPS + 1):
            snapshots = obs.metadata.get('server_snapshots', [])
            state_prompt = f"Step: {step}\nCurrent Infrastructure State:\n{json.dumps(snapshots, indent=2)}\n\nWhat is your next action?"

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state_prompt},
            ]

            print(f"\n--- STEP {step} ---")
            print("Waiting for model decision...")
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
            )
            response_text = completion.choices[0].message.content or ""
            action_dict = extract_json_action(response_text)
            print(f"Model Action: {action_dict.get('action_type', 'UNKNOWN')} on {action_dict.get('target_server_id', 'N/A')}")

            obs = env.step(action_dict)
            print(f"Reward: {obs.reward}")
            print(f"Env Response: {obs.metadata.get('message')}")

            if obs.done:
                print("\n=== EPISODE TERMINATED ===")
                break

    except Exception as e:
        print(f"Connection failed. Make sure you are running 'uv run server' in another terminal! Error: {e}")

if __name__ == "__main__":
    main()
