from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional


@dataclass
class SemanticEntropyContext:
    """
    Context object that replaces Streamlit's session_state for semantic entropy detection.

    This holds the model client and configuration needed for generating responses
    and computing semantic entropy.
    """
    model_client: Any  # OpenAI client or HuggingFace TGI client
    model_name: str  # e.g., "gpt-3.5-turbo", "llama3-instruct"
    one_shot: bool = False
    prompts: Optional[List[str]] = None  # For one-shot random selection
    messages: List[Dict[str, str]] = field(default_factory=list)  # Chat history
