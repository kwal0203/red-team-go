"""
FActScore hallucination detection service.

Based on: FActScore: Fine-grained Atomic Evaluation of Factual Precision
in Long Form Text Generation (Min et al., EMNLP 2023)

This service evaluates the factual precision of generated text by:
1. Breaking the text into sentences
2. Extracting atomic facts from each sentence
3. Checking each fact against provided evidence
4. Computing a score based on the proportion of supported facts
"""

import logging
from typing import Any

import nltk
from src.services.hallucination_detection import detect
from src.utility.atomic_facts import get_atomic_facts

logger = logging.getLogger(__name__)


def service(args: Any) -> Any:
    """
    Evaluate factual precision of generated text using FActScore.

    Args:
        args: Object with the following attributes:
            - summary (str): The generated text to evaluate
            - source (str): The evidence/source text to check facts against
            - domain (str, optional): Domain restriction for fact checking
              (e.g., "genomics", "medicine"). If not provided, checks all facts.

    Returns:
        Dictionary containing:
            - factscore (int): Overall factual precision score (0-100)
            - sentences (list): Individual sentences from the summary
            - scores (list): Per-sentence factual precision scores
            - details (list): Detailed breakdown of facts and their support status
            - summary (str): The original generated summary
            - domain (str|None): The domain restriction used (if any)
    """
    generated_summary = args.summary
    source = args.source
    domain = getattr(args, "domain", None)

    logger.info(
        f"FActScore evaluation: {len(generated_summary)} chars, "
        f"domain={'general' if domain is None else domain}"
    )

    # Split generation into sentences
    sentences = nltk.tokenize.sent_tokenize(generated_summary)
    logger.debug(f"Split into {len(sentences)} sentences")

    # Extract atomic facts from each sentence
    atomic_facts = get_atomic_facts(sentences=sentences)

    # Check facts against evidence
    result_json = detect(
        source=source,
        atomic_facts=atomic_facts,
        domain=domain,
    )
    result_json["summary"] = generated_summary

    logger.info(f"FActScore result: {result_json['factscore']}/100")

    return result_json
