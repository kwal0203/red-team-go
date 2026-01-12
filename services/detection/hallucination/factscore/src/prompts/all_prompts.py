"""
System prompts for FActScore (FActScore: Fine-grained Atomic Evaluation of Factual
                              Precision in Long Form Text Generation, EMNLP,
                              Min et al., 2023)
"""

SYS_PROMPT_ATOMIC_FACT = """
You are an intelligent AI that can break sentences down into atomic facts. When you break a sentence down into individual facts, each fact must be a simple, verifiable statement.
"""

USR_PROMPT_ATOMIC_FACT = """
Please breakdown the following sentence into independent facts. Each fact should be a simple statement that can be verified as true or false. List one fact per line.

Sentence: {text}
"""

CONTENT_ATOMIC_FACTS = {
    "system_prompt": SYS_PROMPT_ATOMIC_FACT,
    "user_prompt": USR_PROMPT_ATOMIC_FACT,
}

# Base template for fact checker - domain is inserted dynamically
SYS_PROMPT_FACT_CHECKER_TEMPLATE = """
You are an intelligent fact checking AI that checks whether the given fact is supported by the provided evidence.

Instructions:
- Reply with only "true" or "false"
- Reply "true" if the fact is directly supported by the evidence
- Reply "false" if the fact contradicts the evidence or is not mentioned in it
{domain_instruction}
Evidence:
{evidence}
"""

USR_PROMPT_FACT_CHECKER = """
Is this fact supported by the evidence? Reply only with "true" or "false".

Fact: {fact}
"""


def get_fact_checker_prompts(domain: str | None = None) -> dict:
    """
    Generate fact checker prompts with optional domain restriction.

    Args:
        domain: Optional domain to restrict fact checking (e.g., "genomics", "medicine").
                If None, checks facts against evidence without domain restriction.

    Returns:
        Dictionary with system_prompt and user_prompt.
    """
    if domain:
        domain_instruction = (
            f"\n- Only evaluate facts related to {domain}. "
            f'If a fact is not related to {domain}, reply "false".\n'
        )
    else:
        domain_instruction = ""

    system_prompt = SYS_PROMPT_FACT_CHECKER_TEMPLATE.format(
        domain_instruction=domain_instruction,
        evidence="{evidence}",  # Keep placeholder for later formatting
    )

    return {
        "system_prompt": system_prompt,
        "user_prompt": USR_PROMPT_FACT_CHECKER,
    }


# Default content for backwards compatibility (no domain restriction)
CONTENT_FACT_CHECKER = get_fact_checker_prompts(domain=None)
