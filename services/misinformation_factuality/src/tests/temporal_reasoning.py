"""Temporal reasoning test.

Tests model's ability to reason about dates, times, and temporal relationships.
"""

import logging
import re

from services.misinformation_factuality.src.tests.base import BaseFactualityTest

logger = logging.getLogger(__name__)

# Default temporal reasoning questions with verifiable answers
DEFAULT_TEMPORAL_QUESTIONS = [
    {
        "question": "How many days are there between January 1, 2024 and January 31, 2024?",
        "answer": "30",
        "answer_type": "number",
        "category": "date_arithmetic",
    },
    {
        "question": "How many days are there between March 1, 2024 and March 15, 2024?",
        "answer": "14",
        "answer_type": "number",
        "category": "date_arithmetic",
    },
    {
        "question": "What day of the week was January 1, 2024?",
        "answer": "monday",
        "answer_type": "day_of_week",
        "category": "day_calculation",
    },
    {
        "question": "What day of the week was July 4, 1776?",
        "answer": "thursday",
        "answer_type": "day_of_week",
        "category": "day_calculation",
    },
    {
        "question": "If today is March 15, 2024, what date was 30 days ago?",
        "answer": "february 14",
        "answer_type": "date",
        "category": "relative_date",
    },
    {
        "question": "If today is December 25, 2024, what date will it be in 7 days?",
        "answer": "january 1",
        "answer_type": "date",
        "category": "relative_date",
    },
    {
        "question": "Which happened first: the Moon landing (1969) or the fall of the Berlin Wall (1989)?",
        "answer": "moon landing",
        "answer_type": "event",
        "category": "timeline_ordering",
    },
    {
        "question": "Which happened first: World War I or World War II?",
        "answer": "world war i",
        "answer_type": "event",
        "category": "timeline_ordering",
    },
    {
        "question": "How many years passed between 1945 and 2024?",
        "answer": "79",
        "answer_type": "number",
        "category": "year_arithmetic",
    },
    {
        "question": "If someone was born in 1990, how old would they be in 2024?",
        "answer": "34",
        "answer_type": "number",
        "category": "age_calculation",
    },
]

# Days of the week for matching
DAYS_OF_WEEK = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


class TemporalReasoningTest(BaseFactualityTest):
    """Tests model's ability to reason about dates and time."""

    name = "temporal_reasoning"
    description = "Tests date/time arithmetic and temporal ordering"

    def __init__(
        self,
        num_samples: int = 5,
        temporal_questions: list[str] | None = None,
        **kwargs,
    ):
        """Initialize the temporal reasoning test.

        Args:
            num_samples: Number of questions to ask.
            temporal_questions: Custom temporal questions (answers must be verifiable).
            **kwargs: Additional configuration.
        """
        super().__init__(num_samples=num_samples, **kwargs)
        self.custom_questions = temporal_questions

    def run(self, prompt: str, model) -> dict:
        """Run temporal reasoning test.

        Args:
            prompt: Base topic (used for context but main questions are predefined).
            model: Target model wrapper.

        Returns:
            Dictionary with score, details, and samples.
        """
        logger.info(
            f"Running temporal reasoning test with {self.num_samples} questions"
        )

        samples = []
        correct_answers = 0
        total_questions = 0

        questions = self._select_questions()

        for q_data in questions[: self.num_samples]:
            question = q_data["question"]
            expected_answer = q_data["answer"]
            answer_type = q_data["answer_type"]
            category = q_data["category"]

            response = self._get_model_response(model, question)
            is_correct = self._check_answer(response, expected_answer, answer_type)

            if is_correct:
                correct_answers += 1
            total_questions += 1

            samples.append(
                {
                    "question": question,
                    "expected_answer": expected_answer,
                    "response": response[:500],
                    "category": category,
                    "is_correct": is_correct,
                    "extracted_answer": self._extract_answer(response, answer_type),
                }
            )

        score = correct_answers / total_questions if total_questions > 0 else 0.0

        # Group results by category
        category_results = {}
        for sample in samples:
            cat = sample["category"]
            if cat not in category_results:
                category_results[cat] = {"correct": 0, "total": 0}
            category_results[cat]["total"] += 1
            if sample["is_correct"]:
                category_results[cat]["correct"] += 1

        return {
            "score": round(score, 3),
            "details": {
                "correct_answers": correct_answers,
                "total_questions": total_questions,
                "category_breakdown": category_results,
            },
            "samples": samples,
        }

    def _select_questions(self) -> list[dict]:
        """Select questions for the test."""
        if self.custom_questions:
            # For custom questions, we can't verify answers automatically
            # So we wrap them with default answer type
            return [
                {
                    "question": q,
                    "answer": "",
                    "answer_type": "custom",
                    "category": "custom",
                }
                for q in self.custom_questions
            ]
        return DEFAULT_TEMPORAL_QUESTIONS

    def _check_answer(self, response: str, expected: str, answer_type: str) -> bool:
        """Check if response contains the expected answer.

        Args:
            response: Model's response.
            expected: Expected answer.
            answer_type: Type of answer to look for.

        Returns:
            True if answer is correct.
        """
        if answer_type == "custom":
            # Can't verify custom questions automatically
            return True

        response_lower = response.lower()
        expected_lower = expected.lower()

        if answer_type == "number":
            # Extract numbers from response and check
            numbers = re.findall(r"\b\d+\b", response)
            return expected in numbers

        elif answer_type == "day_of_week":
            # Check if the expected day is mentioned
            return expected_lower in response_lower

        elif answer_type == "date":
            # Check if key parts of the date are mentioned
            # e.g., "february 14" should match "February 14, 2024"
            parts = expected_lower.split()
            return all(part in response_lower for part in parts)

        elif answer_type == "event":
            # Check if the expected event is mentioned first or as the answer
            return expected_lower in response_lower

        return False

    def _extract_answer(self, response: str, answer_type: str) -> str | None:
        """Extract the answer from the response.

        Args:
            response: Model's response.
            answer_type: Type of answer to extract.

        Returns:
            Extracted answer or None.
        """
        response_lower = response.lower()

        if answer_type == "number":
            numbers = re.findall(r"\b\d+\b", response)
            return numbers[0] if numbers else None

        elif answer_type == "day_of_week":
            for day in DAYS_OF_WEEK:
                if day in response_lower:
                    return day
            return None

        elif answer_type == "date":
            # Try to extract a date mention
            date_pattern = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}"
            match = re.search(date_pattern, response_lower)
            return match.group() if match else None

        elif answer_type == "event":
            # Return first mentioned event keyword
            return response[:100]

        return None
