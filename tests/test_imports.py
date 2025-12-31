"""Verify all main dependencies can be imported correctly."""


def test_import_torch():
    """Test PyTorch can be imported."""
    import torch

    assert torch.__version__ is not None


def test_import_openai():
    """Test OpenAI SDK can be imported."""
    import openai

    assert openai.__version__ is not None


def test_import_fastapi():
    """Test FastAPI can be imported."""
    import fastapi

    assert fastapi.__version__ is not None


def test_import_pydantic():
    """Test Pydantic can be imported."""
    import pydantic

    assert pydantic.__version__ is not None


def test_import_transformers():
    """Test Transformers can be imported."""
    import transformers

    assert transformers.__version__ is not None


def test_import_nltk():
    """Test NLTK can be imported."""
    import nltk

    assert nltk.__version__ is not None


def test_import_app():
    """Test the main FastAPI app can be imported."""
    from main import app

    assert app is not None


def test_import_utils():
    """Test utils module can be imported."""
    from utils import config, models, utils

    assert config is not None
    assert models is not None
    assert utils is not None
