import pandas as pd


class PromptInjectionData:
    """
    Class that ingests information in the prompt database (currently a .csv file)

    Attributes:
        prompt_injection_df (pd.Dataframe): TODO.

    Methods:
        get_prompts_all: TODO.
        get_prompts_name: TODO.
        get_prompts_type: TODO.
        get_attack_names: TODO.
        get_attack_types: TODO.
    """

    def __init__(self, path: str) -> None:
        """
        Initializes PromptInjectionData with following attributes.

        Args:
            prompt_injection_df (pd.Dataframe): TODO.
        """
        self.prompt_injection_df = pd.read_csv(filepath_or_buffer=path)

    def get_prompts_all(self) -> list[str]:
        """
        Get all prompts in prompt database.
        """
        return self.prompt_injection_df.prompt.tolist()

    def get_prompts_name(self, attack_name: str) -> list[str]:
        """
        Get all prompts of a given attack from prompt database.

        Args:
            attack_name (str): Name of attack (i.e. DAN 9.0, 'nevermind')
        """
        return self.prompt_injection_df[
            self.prompt_injection_df.name == attack_name
        ].prompt.tolist()

    def get_prompts_type(self, attack_type: str) -> list[str]:
        """
        Get all prompts of a given type form prompt database.

        Args:
            attack_type (str): Type of attack (i.e. hijacking, jailbreak)
        """
        return self.prompt_injection_df[
            self.prompt_injection_df.group == attack_type
        ].prompt.tolist()

    def get_attack_names(self) -> list[str]:
        """
        Get the name of all attacks in the prompt database.
        """
        return list(set(self.prompt_injection_df.name))

    def get_attack_types(self) -> list[str]:
        """
        Get all attack types of prompts in the database.
        """
        return list(set(self.prompt_injection_df.group))
