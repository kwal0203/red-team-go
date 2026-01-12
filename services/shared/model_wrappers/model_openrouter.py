import os

import openai

from services.shared.model_wrappers.base_model_remote import APIModel
from utils.config import (
    get_openrouter_base_url,
    get_openrouter_key,
    get_openrouter_model_name,
)


class APIModelOpenRouter(APIModel):
    """
    Wrapper for OpenRouter-compatible OpenAI API.
    Uses OpenAI client pointed at OpenRouter base URL.
    """

    def __init__(
        self,
        name: str | None = "openrouter_api_model",
        description: str | None = "OpenRouter model wrapper",
        model_name: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        api_key = get_openrouter_key()
        # Ensure trailing slash so paths are correct (e.g., .../v1/chat/completions)
        base_url = get_openrouter_base_url().rstrip("/") + "/"
        default_headers = {
            # Recommended by OpenRouter to help with analytics; optional but harmless
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "RedTeamGo"),
        }
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
        )
        self.model_name = model_name or get_openrouter_model_name()

    def _model_predict(self, inputs: list[str]) -> list[str]:
        """
        Sends preprocessed inputs to the OpenRouter API and retrieves responses.

        Args:
            inputs: List of prompts.

        Returns:
            List of responses from the OpenRouter backend.
        """
        responses = []
        for input_text in inputs:
            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": input_text}],
                    model=self.model_name,
                    max_tokens=150,
                    temperature=0.7,
                    timeout=20,
                )
                responses.append(response.choices[0].message.content)
            except Exception as e:
                responses.append(f"Error: {str(e)}")
        return responses
