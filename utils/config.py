import os

from dotenv import load_dotenv

load_dotenv()


def get_openai_key():
    return os.getenv("API_KEY_OPENAI")


def get_device():
    return os.getenv("DEVICE", "cpu")


def get_hf_key():
    return os.getenv("HF_TOKEN")


def get_api_keys() -> str | None:
    """
    Get comma-separated list of valid API keys.

    Example: REDTEAM_API_KEYS="key1,key2,key3"
    Returns None if not configured (dev mode).
    """
    return os.getenv("REDTEAM_API_KEYS")
