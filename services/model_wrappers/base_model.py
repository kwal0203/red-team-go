from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelMeta:
    name: str | None
    description: str | None


class BaseModel:
    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """
        Initialize a new instance of the BaseModel class.

        Parameters:
            name (Optional[str]): Name of the model. If not provided, defaults
                to the class name.
            description (Optional[str]): Description of the model's task.
                Mandatory for non-langchain text_generation models.

        Raises:
            ValueError: If an invalid model_type value is provided.
        """

        self.meta = ModelMeta(
            name=name if name is not None else self.__class__.__name__,
            description=description if description is not None else "default",
        )

    @property
    def name(self):
        if self.meta.name is not None:
            return self.meta.name
        else:
            return self.__class__.__name__

    @property
    def description(self):
        return self.meta.description


class WrapperModel(BaseModel, ABC):
    """
    A wrapper base class for all redteam models.

    Attributes:
        name (Optional[str]): The name of the wrapper model. Defaults to
            "my_model".
        description (Optional[str]): A description of the model being wrapped.
            Defaults to "Large language model".
    """

    def __init__(
        self,
        name: str | None = "my_model",
        description: str | None = "Large language model",
    ) -> None:
        """
        Initializes the WrapperModel with the given name and description.

        Args:
            name (Optional[str]): A name for the wrapper model. Defaults to
                "my_model".
            description (Optional[str]): A description of the model being
                wrapped. Defaults to "Large language model".
        """
        super().__init__(name=name, description=description)

    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """
        Abstract method for preprocessing input data.

        Args:
            data: The input data to be preprocessed.

        Returns:
            The preprocessed data.
        """
        pass

    @abstractmethod
    def postprocess(self, data: Any) -> Any:
        """
        Abstract method for postprocessing the model's output.

        Args:
            data: The output data from the model to be postprocessed.

        Returns:
            The postprocessed data.
        """
        pass

    @abstractmethod
    def model_predict(self, data: Any) -> Any:
        """
        Abstract method for generating predictions using the wrapped model.

        Args:
            data: The input data for the model.

        Returns:
            The model's predictions.
        """
        pass
