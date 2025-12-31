from typing import Any

from openai import OpenAI

from utils.config import get_openai_key


class OpenAIModel:
    """
    This class provides a wrapper for the OpenAI API.

    Attributes:
        prompts (Dict): A dictionary containing system and user prompts.
        fact_checker (Bool): Flag to specify whether model is required to check facts.
        name (str): The OpenAI API name of the model.
        batch_size (int): The batch size used for inference. Default to 1.

    Methods:
        preprocess: Preprocesses input prompts to be compliant with OpenAI API.
        postprocess: Postprocess the response from OpenAI API.
        model_predict: Query the OpenAI API.
        _model_predict: Helper function for obtaining generated response from input prompts..
    """

    def __init__(
        self,
        prompts: dict = None,
        fact_checker: bool = False,
        name: str | None = None,
        batch_size: int | None = 1,
    ) -> None:
        """
        Initializes the OpenAIModel class.

        Args:
            prompts (Dict): A dictionary containing system and user prompts.
            fact_checker (Bool): Flag to specify whether model is required to check facts.
            name (str): The OpenAI API name of the model.
            batch_size (int): The batch size used for inference. Default to 1.
        """
        self.model = OpenAI(api_key=get_openai_key())
        self.prompts = prompts
        self.fact_checker = fact_checker
        self.name = name
        self.batch_size = batch_size

    def preprocess(self, data: str) -> list[dict]:
        """
        Preprocesses input prompts to be compliant with OpenAI API.

        Args:
            data (str): Sentence that user wants broken into atomic facts.
        """

        if self.fact_checker:
            system_prompt = self.prompts["system_prompt"].format(
                evidence=f"{data['evidence']}"
            )
            user_prompt = self.prompts["user_prompt"].format(fact=f"{data['fact']}")
        else:
            system_prompt = self.prompts["system_prompt"]
            user_prompt = self.prompts["user_prompt"].format(text=f"{data}")

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    def postprocess(self, data):
        """
        Process the response from OpenAI API.

        Args:
            data (Any): Output obtained from OpenAI API call.
        """
        return data

    def model_predict(self, data: str) -> list[Any]:
        """
        Query the OpenAI API.

        Args:
            data (str): data (str): Sentence that user wants broken into atomic facts.
        """
        inputs = self.preprocess(data)
        responses = self._model_predict([inputs])
        return responses

    def _model_predict(self, inputs: list[dict]) -> list[Any]:
        """
        Helper function for obtaining generated response from input prompts.

        Args:
            inputs (List[Dict]): List of sentences to turn into atomic facts.
        """

        responses = []
        for message in inputs:
            response = self.model.chat.completions.create(
                model=self.name, messages=message
            )
            responses.append(response)

        return responses
