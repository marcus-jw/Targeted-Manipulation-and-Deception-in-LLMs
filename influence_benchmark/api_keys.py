import os

from dotenv import load_dotenv

from influence_benchmark.root import PROJECT_ROOT

is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"

LOADED_DOTENV = load_dotenv(PROJECT_ROOT / ".env")  # Can import this var in other files if access to API keys is needed
if not is_github_actions:
    assert LOADED_DOTENV, ".env file not found in influence_benchmark/.env"
