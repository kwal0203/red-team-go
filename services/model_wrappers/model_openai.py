import openai

from services.model_wrappers.base_model_remote import APIModel
from utils.config import get_openai_key, get_openai_model_name


class APIModelOpenai(APIModel):
    """
    A wrapper model class for interacting with the OpenAI API specifically.
    Inherits from APIModel and overrides _model_predict for OpenAI-specific behavior.
    """

    def __init__(
        self,
        name: str | None = "openai_api_model",
        description: str | None = "OpenAI model wrapper",
        model_name: str | None = None,
    ) -> None:
        """
        Initializes the OpenAI API model with the given name and description.
        """
        super().__init__(name=name, description=description)
        openai.api_key = get_openai_key()
        self.client = openai
        self.model_name = model_name or get_openai_model_name()

    def _model_predict(self, inputs: list[str]) -> list[str]:
        """
        Sends preprocessed inputs to the OpenAI API and retrieves responses.

        Args:
            inputs (List[str]): A list of preprocessed input strings.

        Returns:
            List[str]: A list of responses from the OpenAI API.
        """

        responses = []
        for input_text in inputs:
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        # {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input_text},
                    ],
                    model=self.model_name,
                    max_tokens=150,
                    temperature=0.7,
                )
                responses.append(response.choices[0].message.content)
            except Exception as e:
                responses.append(f"Error: {str(e)}")
        return responses
