import os
from dotenv.main import load_dotenv

# Load .env from the SearchAgent-B root (one level up from services/)
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)
