from services.shared.model_wrappers.base_model import WrapperModel


class APIModel(WrapperModel):
    """
    A wrapper model class for interacting with a remote large language model via API.

    Attributes:
        name (Optional[str]): The name of the API model. Defaults to "my_api_model".
        description (Optional[str]): A description of the model being wrapped. Defaults to "Remote large language model".
        openai_key (str): The API key used for authentication with the OpenAI endpoint.
    """

    def __init__(
        self,
        name: str | None = "my_api_model",
        description: str | None = "Remote large language model",
    ) -> None:
        """
        Initializes the APIModel with the given name and description, and sets up the OpenAI API key.

        Args:
            name (Optional[str]): The name of the API model. Defaults to "my_api_model".
            description (Optional[str]): A description of the model being wrapped. Defaults to "Remote large language model".
        """
        super().__init__(name=name, description=description)

    def preprocess(self, data):
        """
        Preprocesses input data before sending it to the model.

        Args:
            data: The input data to be preprocessed.

        Returns:
            The preprocessed data as a list.
        """
        # Ensure data is a list to prevent iterating over characters
        if isinstance(data, str):
            return [data]
        return data

    def postprocess(self, data):
        """
        Postprocesses the model's response to prepare it for output.

        Args:
            data: The raw response from the model.

        Returns:
            The processed response.
        """
        return data

    def model_predict(self, data):
        """
        Generates predictions from the model for the given input data.

        Args:
            data: The input data for the model.

        Returns:
            The model's predictions.
        """
        input = self.preprocess(data=data)
        response = self._model_predict(inputs=input)
        output = self.postprocess(data=response)
        return output

    def _model_predict(self, inputs: list[str]) -> list[str]:
        """
        Sends preprocessed inputs to the remote model and retrieves responses.

        Args:
            inputs (List[str]): A list of preprocessed input strings.

        Returns:
            List[str]: A list of responses from the remote model.
        """
        # TODO: set this up so it queries the provided FastAPI endpoint
        raise NotImplementedError(
            "Subclasses must implement _model_predict to query the remote model"
        )
