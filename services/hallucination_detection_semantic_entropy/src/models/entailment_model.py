from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import torch
import torch.nn.functional as F

# Determine device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EntailmentDeberta:
    """
    DeBERTa-based entailment model for semantic equivalence checking.

    Uses microsoft/deberta-v2-xlarge-mnli to determine if two texts
    are semantically equivalent (entail each other).
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        ).to(device)

    def check_implication(self, text1: str, text2: str, *args, **kwargs) -> int:
        """
        Check if text1 implies text2.

        Args:
            text1: The premise text.
            text2: The hypothesis text.

        Returns:
            int: 0 = contradiction, 1 = neutral, 2 = entailment
        """
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Deberta-mnli returns classes: 0=contradiction, 1=neutral, 2=entailment
        largest_index = torch.argmax(F.softmax(logits, dim=1))
        prediction = largest_index.cpu().item()
        return prediction
