import openai

from services.model_wrappers.base_model_remote import APIModel
from utils.config import get_hf_key


class APIModelHuggingFace(APIModel):
    """
    A wrapper class for interacting with HuggingFace models using TGI
    (https://github.com/huggingface/text-generation-inference). TGI implements
    the OpenAI API for HuggingFace models.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8995/v1",
        name: str | None = "huggingface_tgi_model",
        description: str | None = "TGI model wrapper",
    ) -> None:
        """
        Initializes a HuggingFace model using TGI with the given name and description.
        """
        super().__init__(name=name, description=description)
        self.hf_token = get_hf_key()
        self.client = openai.OpenAI(base_url=base_url, api_key=self.hf_token)

    def _model_predict(self, inputs: list[str]) -> list[str]:
        """
        Sends preprocessed inputs to HuggingFace model and retrieves responses.

        Args:
            inputs (List[str]): A list of preprocessed input strings.

        Returns:
            List[str]: A list of responses from the HuggingFace model.
        """

        responses = []
        for input_text in inputs:
            try:
                response = self.client.chat.completions.create(
                    model="tgi",
                    messages=[
                        # {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input_text},
                    ],
                    stream=False,
                )

                content = response.choices[0].message.content
                responses.append(content)
            except Exception as e:
                responses.append(f"Error in APIModelHuggingFace: {str(e)}")

        return responses
