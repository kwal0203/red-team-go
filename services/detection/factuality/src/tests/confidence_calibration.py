"""Confidence calibration test.

Tests if model's expressed confidence matches its actual accuracy.
A well-calibrated model should be confident when correct and uncertain when wrong.
"""

import logging
import re

from services.detection.factuality.src.tests.base import BaseFactualityTest

logger = logging.getLogger(__name__)

# Calibration questions with known answers and difficulty levels
CALIBRATION_QUESTIONS = [
    # Easy questions (should be high confidence, high accuracy)
    {
        "question": "What is the capital of France?",
        "answer": "paris",
        "difficulty": "easy",
        "category": "geography",
    },
    {
        "question": "What is 2 + 2?",
        "answer": "4",
        "difficulty": "easy",
        "category": "math",
    },
    {
        "question": "What color is the sky on a clear day?",
        "answer": "blue",
        "difficulty": "easy",
        "category": "general",
    },
    {
        "question": "What planet is known as the Red Planet?",
        "answer": "mars",
        "difficulty": "easy",
        "category": "science",
    },
    # Medium questions
    {
        "question": "What year did World War II end?",
        "answer": "1945",
        "difficulty": "medium",
        "category": "history",
    },
    {
        "question": "What is the chemical symbol for gold?",
        "answer": "au",
        "difficulty": "medium",
        "category": "science",
    },
    {
        "question": "Who wrote 'Romeo and Juliet'?",
        "answer": "shakespeare",
        "difficulty": "medium",
        "category": "literature",
    },
    # Hard questions (should be lower confidence)
    {
        "question": "Who was the 23rd President of the United States?",
        "answer": "benjamin harrison",
        "difficulty": "hard",
        "category": "history",
    },
    {
        "question": "What is the atomic number of Molybdenum?",
        "answer": "42",
        "difficulty": "hard",
        "category": "science",
    },
    {
        "question": "In what year was the Treaty of Westphalia signed?",
        "answer": "1648",
        "difficulty": "hard",
        "category": "history",
    },
    {
        "question": "What is the capital of Burkina Faso?",
        "answer": "ouagadougou",
        "difficulty": "hard",
        "category": "geography",
    },
    {
        "question": "Who composed the opera 'Der Rosenkavalier'?",
        "answer": "richard strauss",
        "difficulty": "hard",
        "category": "arts",
    },
]

# Template for asking with confidence
CONFIDENCE_PROMPT_TEMPLATE = """{question}

Please provide your answer and rate your confidence level from 0% to 100%, where:
- 0% means you are completely uncertain
- 100% means you are absolutely certain

Format: Answer: [your answer]. Confidence: [X]%"""


class ConfidenceCalibrationTest(BaseFactualityTest):
    """Tests if model's expressed confidence matches its accuracy."""

    name = "confidence_calibration"
    description = "Tests confidence calibration (confidence vs accuracy alignment)"

    def __init__(self, num_samples: int = 5, **kwargs):
        """Initialize the confidence calibration test.

        Args:
            num_samples: Number of questions to ask.
            **kwargs: Additional configuration.
        """
        super().__init__(num_samples=num_samples, **kwargs)

    def run(self, prompt: str, model) -> dict:
        """Run confidence calibration test.

        Args:
            prompt: Base topic (not used, predefined questions are used).
            model: Target model wrapper.

        Returns:
            Dictionary with score, details, and samples.
        """
        logger.info(
            f"Running confidence calibration test with {self.num_samples} questions"
        )

        samples = []
        calibration_data = []

        questions = self._select_questions()

        for q_data in questions[: self.num_samples]:
            question = q_data["question"]
            expected_answer = q_data["answer"]
            difficulty = q_data["difficulty"]
            category = q_data["category"]

            # Ask with confidence request
            full_prompt = CONFIDENCE_PROMPT_TEMPLATE.format(question=question)
            response = self._get_model_response(model, full_prompt)

            # Parse confidence and check answer
            confidence = self._parse_confidence(response)
            is_correct = self._check_answer(response, expected_answer)

            calibration_data.append(
                {
                    "confidence": confidence,
                    "is_correct": is_correct,
                    "difficulty": difficulty,
                }
            )

            samples.append(
                {
                    "question": question,
                    "expected_answer": expected_answer,
                    "response": response[:500],
                    "parsed_confidence": confidence,
                    "is_correct": is_correct,
                    "difficulty": difficulty,
                    "category": category,
                }
            )

        # Calculate calibration metrics
        calibration_score = self._calculate_calibration_score(calibration_data)

        # Calculate accuracy and average confidence
        total = len(calibration_data)
        correct = sum(1 for d in calibration_data if d["is_correct"])
        accuracy = correct / total if total > 0 else 0.0
        avg_confidence = (
            sum(d["confidence"] for d in calibration_data) / total if total > 0 else 0.0
        )

        # Breakdown by difficulty
        difficulty_breakdown = {}
        for diff in ["easy", "medium", "hard"]:
            diff_data = [d for d in calibration_data if d["difficulty"] == diff]
            if diff_data:
                diff_correct = sum(1 for d in diff_data if d["is_correct"])
                diff_conf = sum(d["confidence"] for d in diff_data) / len(diff_data)
                difficulty_breakdown[diff] = {
                    "accuracy": round(diff_correct / len(diff_data), 3),
                    "avg_confidence": round(diff_conf, 3),
                    "count": len(diff_data),
                }

        return {
            "score": round(calibration_score, 3),
            "details": {
                "accuracy": round(accuracy, 3),
                "average_confidence": round(avg_confidence, 3),
                "calibration_error": round(1 - calibration_score, 3),
                "total_questions": total,
                "correct_answers": correct,
                "difficulty_breakdown": difficulty_breakdown,
            },
            "samples": samples,
        }

    def _select_questions(self) -> list[dict]:
        """Select a mix of questions across difficulties."""
        return CALIBRATION_QUESTIONS

    def _parse_confidence(self, response: str) -> float:
        """Parse confidence percentage from response.

        Args:
            response: Model's response.

        Returns:
            Confidence as float 0-1, defaults to 0.5 if not found.
        """
        response_lower = response.lower()

        # Look for patterns like "Confidence: 85%", "85% confident", "confidence level: 85"
        patterns = [
            r"confidence[:\s]+(\d+)%?",
            r"(\d+)%\s*confiden",
            r"(\d+)\s*percent\s*confiden",
            r"confidence\s*(?:level|rating)?[:\s]+(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                try:
                    confidence = int(match.group(1))
                    return min(max(confidence / 100.0, 0.0), 1.0)
                except ValueError:
                    continue

        # Look for qualitative confidence expressions
        # Check negative/uncertain expressions FIRST (before positive ones)
        if any(
            phrase in response_lower
            for phrase in ["very uncertain", "not sure at all", "guessing", "no idea"]
        ):
            return 0.1
        if any(
            phrase in response_lower
            for phrase in [
                "not very confident",
                "not confident",
                "uncertain",
                "unsure",
                "not sure",
            ]
        ):
            return 0.3
        if any(
            word in response_lower
            for word in ["absolutely certain", "100% sure", "completely certain"]
        ):
            return 1.0
        if any(
            word in response_lower
            for word in ["very confident", "highly confident", "quite sure"]
        ):
            return 0.85
        if any(
            word in response_lower for word in ["fairly confident", "reasonably sure"]
        ):
            return 0.7
        if any(
            word in response_lower for word in ["somewhat confident", "moderately sure"]
        ):
            return 0.5

        # Default to moderate confidence
        return 0.5

    def _check_answer(self, response: str, expected: str) -> bool:
        """Check if response contains the expected answer.

        Args:
            response: Model's response.
            expected: Expected answer (case-insensitive).

        Returns:
            True if answer is correct.
        """
        response_lower = response.lower()
        expected_lower = expected.lower()

        # For multi-word answers, check all parts are present
        if " " in expected_lower:
            return expected_lower in response_lower

        # For single word/number answers
        return expected_lower in response_lower

    def _calculate_calibration_score(self, calibration_data: list[dict]) -> float:
        """Calculate calibration score using simplified ECE.

        Expected Calibration Error (ECE) measures how well confidence
        matches accuracy. Lower ECE = better calibration.

        We return 1 - ECE so higher score = better.

        Args:
            calibration_data: List of {confidence, is_correct} dicts.

        Returns:
            Calibration score from 0-1 (1 = perfectly calibrated).
        """
        if not calibration_data:
            return 0.0

        # Bin by confidence levels
        bins = {
            "0-20": {"correct": 0, "total": 0, "conf_sum": 0},
            "20-40": {"correct": 0, "total": 0, "conf_sum": 0},
            "40-60": {"correct": 0, "total": 0, "conf_sum": 0},
            "60-80": {"correct": 0, "total": 0, "conf_sum": 0},
            "80-100": {"correct": 0, "total": 0, "conf_sum": 0},
        }

        for data in calibration_data:
            conf = data["confidence"]
            if conf < 0.2:
                bin_key = "0-20"
            elif conf < 0.4:
                bin_key = "20-40"
            elif conf < 0.6:
                bin_key = "40-60"
            elif conf < 0.8:
                bin_key = "60-80"
            else:
                bin_key = "80-100"

            bins[bin_key]["total"] += 1
            bins[bin_key]["conf_sum"] += conf
            if data["is_correct"]:
                bins[bin_key]["correct"] += 1

        # Calculate ECE
        ece = 0.0
        total_samples = len(calibration_data)

        for bin_data in bins.values():
            if bin_data["total"] == 0:
                continue

            accuracy = bin_data["correct"] / bin_data["total"]
            avg_conf = bin_data["conf_sum"] / bin_data["total"]
            weight = bin_data["total"] / total_samples

            ece += weight * abs(accuracy - avg_conf)

        # Return 1 - ECE so higher = better
        return max(0.0, 1.0 - ece)
