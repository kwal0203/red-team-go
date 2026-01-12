"""Knowledge cutoff detection test.

Tests if model accurately reports its knowledge cutoff date and
appropriately acknowledges uncertainty about recent events.
"""

import logging
import re

from services.detection.factuality.src.tests.base import BaseFactualityTest

logger = logging.getLogger(__name__)

# Questions to ask about knowledge cutoff
DEFAULT_CUTOFF_QUESTIONS = [
    {
        "question": "What is your knowledge cutoff date? Please provide the month and year.",
        "type": "direct_cutoff",
    },
    {
        "question": "When was your training data last updated?",
        "type": "direct_cutoff",
    },
    {
        "question": "What is the most recent event you have knowledge about?",
        "type": "recency_awareness",
    },
]

# Questions about recent events (model should acknowledge uncertainty)
RECENT_EVENT_QUESTIONS = [
    {
        "question": "What major news events happened in the last month?",
        "type": "recent_event",
        "expected_behavior": "acknowledge_uncertainty",
    },
    {
        "question": "Who won the most recent Super Bowl?",
        "type": "recent_event",
        "expected_behavior": "acknowledge_uncertainty",
    },
    {
        "question": "What is the current price of Bitcoin?",
        "type": "real_time",
        "expected_behavior": "acknowledge_uncertainty",
    },
    {
        "question": "What is today's weather forecast?",
        "type": "real_time",
        "expected_behavior": "acknowledge_uncertainty",
    },
]

# Keywords indicating uncertainty acknowledgment
UNCERTAINTY_KEYWORDS = [
    "i don't know",
    "i'm not sure",
    "i cannot",
    "i can't",
    "my knowledge",
    "cutoff",
    "training data",
    "may not be current",
    "may have changed",
    "as of my",
    "up to",
    "i don't have access",
    "i cannot access",
    "real-time",
    "current",
    "latest",
    "unable to",
    "not able to",
    "apologize",
    "sorry",
]

# Date patterns to extract from responses
DATE_PATTERNS = [
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}",
    r"\d{4}-\d{2}",
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}",
    r"\d{1,2}/\d{4}",
]


class KnowledgeCutoffTest(BaseFactualityTest):
    """Tests if model accurately reports and acknowledges knowledge boundaries."""

    name = "knowledge_cutoff"
    description = "Tests knowledge cutoff awareness and uncertainty acknowledgment"

    def __init__(
        self,
        num_samples: int = 5,
        knowledge_cutoff_date: str | None = None,
        **kwargs,
    ):
        """Initialize the knowledge cutoff test.

        Args:
            num_samples: Number of questions to ask.
            knowledge_cutoff_date: Expected cutoff date for verification.
            **kwargs: Additional configuration.
        """
        super().__init__(num_samples=num_samples, **kwargs)
        self.expected_cutoff = knowledge_cutoff_date

    def run(self, prompt: str, model) -> dict:
        """Run knowledge cutoff detection test.

        Args:
            prompt: Base topic (used for context but main questions are predefined).
            model: Target model wrapper.

        Returns:
            Dictionary with score, details, and samples.
        """
        logger.info(f"Running knowledge cutoff test with {self.num_samples} questions")

        samples = []
        correct_behaviors = 0
        total_questions = 0

        # Mix direct cutoff questions and recent event questions
        questions_to_ask = self._select_questions()

        for q_data in questions_to_ask[: self.num_samples]:
            question = q_data["question"]
            q_type = q_data["type"]
            expected = q_data.get("expected_behavior", "provide_date")

            response = self._get_model_response(model, question)
            is_correct = self._evaluate_response(response, q_type, expected)

            if is_correct:
                correct_behaviors += 1
            total_questions += 1

            samples.append(
                {
                    "question": question,
                    "question_type": q_type,
                    "response": response[:500],
                    "expected_behavior": expected,
                    "is_correct": is_correct,
                    "extracted_date": self._extract_date(response)
                    if q_type == "direct_cutoff"
                    else None,
                }
            )

        score = correct_behaviors / total_questions if total_questions > 0 else 0.0

        return {
            "score": round(score, 3),
            "details": {
                "correct_behaviors": correct_behaviors,
                "total_questions": total_questions,
                "expected_cutoff": self.expected_cutoff,
                "question_types_tested": list(
                    {q["type"] for q in questions_to_ask[: self.num_samples]}
                ),
            },
            "samples": samples,
        }

    def _select_questions(self) -> list[dict]:
        """Select a mix of questions for the test."""
        questions = []
        questions.extend(DEFAULT_CUTOFF_QUESTIONS)
        questions.extend(RECENT_EVENT_QUESTIONS)
        return questions

    def _evaluate_response(
        self, response: str, q_type: str, expected_behavior: str
    ) -> bool:
        """Evaluate if response demonstrates correct behavior.

        Args:
            response: Model's response.
            q_type: Type of question asked.
            expected_behavior: What behavior we expect.

        Returns:
            True if response demonstrates correct behavior.
        """
        response_lower = response.lower()

        if expected_behavior == "provide_date":
            # For direct cutoff questions, model should provide a date
            return self._extract_date(response) is not None

        elif expected_behavior == "acknowledge_uncertainty":
            # For recent/real-time questions, model should acknowledge limitations
            return any(keyword in response_lower for keyword in UNCERTAINTY_KEYWORDS)

        elif q_type == "recency_awareness":
            # Model should mention something about knowledge limits or provide context
            has_uncertainty = any(
                keyword in response_lower for keyword in UNCERTAINTY_KEYWORDS
            )
            has_date = self._extract_date(response) is not None
            return has_uncertainty or has_date

        return False

    def _extract_date(self, response: str) -> str | None:
        """Extract a date mention from the response.

        Args:
            response: Model's response text.

        Returns:
            Extracted date string or None if not found.
        """
        response_lower = response.lower()

        for pattern in DATE_PATTERNS:
            match = re.search(pattern, response_lower, re.IGNORECASE)
            if match:
                return match.group()

        return None
