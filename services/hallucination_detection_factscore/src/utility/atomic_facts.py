import re

from ..models.open_ai import OpenAIModel
from ..prompts.all_prompts import CONTENT_ATOMIC_FACTS


def get_atomic_facts(sentences: list[str]) -> list[dict]:
    """
    Helper function for hallucination detection. Currently uses FActScore based
    on OpenAI API calls. Sentence is in text form, not tokenized.

    FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form
    Text Generation, EMNLP, Min et al., 2023

    We break out a generation automatically by splitting a generation into
    sentences, and feeding each sentence to InstructGPT (text-davinci-003)
    with a series of instructions to further break it down to a series of
    atomic facts. The prompt to InstructGPT is provided in Table 15. Outputs
    from InstructGPT are used (1) to human experts for revision (Section 3.3)
    and (2) for model-based evaluators (Section 4). We find human experts split
    and merged atomic facts from InstructGPT for 18% and 34% of the cases,
    respectively.
    """
    model = OpenAIModel(
        prompts=CONTENT_ATOMIC_FACTS,
        fact_checker=False,
        name="gpt-3.5-turbo",
        # model="atomic_fact",
    )

    # Data structure with key: sentence, value: atomic facts
    atomic_facts = {}

    # TODO: Add some ICL examples in prompt before looping
    for sentence in sentences:
        facts = model.model_predict(data=sentence)
        atomic_facts[sentence] = facts

    results = []
    for key, val in atomic_facts.items():
        facts = val[0].to_dict()["choices"][0]["message"]["content"]
        _facts = list(facts.split("\n"))
        _facts = [re.sub(r"^[^a-zA-Z]+", "", sentence) for sentence in _facts]
        results.append({"sentence": key, "facts": _facts, "supported": []})

    return results
