"""Configuration required by the local CLI and RunPod client."""
import os
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "3wjmfk65eoyfd8")

MAX_WIDTH = 1024
