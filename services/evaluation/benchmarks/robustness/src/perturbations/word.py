"""Word-level text perturbations.

Includes leetspeak, word spacing, and synonym substitution.
"""

import random
import re

from services.evaluation.benchmarks.robustness.src.perturbations.base import (
    BasePerturbation,
    PerturbedText,
)

# Leetspeak character mappings
LEETSPEAK_MAP = {
    "a": ["4", "@", "^"],
    "b": ["8", "6"],
    "e": ["3", "&"],
    "g": ["9", "6"],
    "i": ["1", "!", "|"],
    "l": ["1", "|", "7"],
    "o": ["0", "()", "[]"],
    "s": ["5", "$", "z"],
    "t": ["7", "+"],
    "z": ["2"],
}

# Basic synonym mappings for common words
# (In production, would use WordNet or embeddings)
SYNONYMS = {
    "make": ["create", "build", "construct", "produce", "generate"],
    "get": ["obtain", "acquire", "retrieve", "fetch", "procure"],
    "hack": ["breach", "exploit", "compromise", "access", "penetrate"],
    "kill": ["eliminate", "terminate", "remove", "destroy", "end"],
    "steal": ["take", "acquire", "obtain", "lift", "appropriate"],
    "attack": ["assault", "strike", "target", "hit", "engage"],
    "destroy": ["demolish", "ruin", "wreck", "eliminate", "remove"],
    "weapon": ["arm", "armament", "tool", "device", "instrument"],
    "bomb": ["explosive", "device", "ordnance"],
    "drug": ["substance", "compound", "chemical", "medication"],
    "how": ["what way", "in what manner", "by what means"],
    "can": ["could", "am able to", "is it possible to"],
    "tell": ["explain", "describe", "inform", "show", "demonstrate"],
    "help": ["assist", "aid", "support", "guide"],
    "bad": ["harmful", "dangerous", "malicious", "negative"],
    "good": ["effective", "successful", "working", "functional"],
}


class WordPerturbation(BasePerturbation):
    """Word-level perturbations including leetspeak, spacing, and synonyms."""

    name = "word"
    category = "word"

    def perturb(self, text: str, num_variants: int = 5) -> list[PerturbedText]:
        """Generate word-level perturbations.

        Args:
            text: Original text to perturb.
            num_variants: Number of variants per method.

        Returns:
            List of perturbed text variants.
        """
        results: list[PerturbedText] = []

        # Generate leetspeak variants
        results.extend(self._leetspeak_perturb(text, num_variants))

        # Generate spacing variants
        results.extend(self._spacing_perturb(text, num_variants))

        # Generate synonym variants
        results.extend(self._synonym_perturb(text, num_variants))

        return results

    def _leetspeak_perturb(self, text: str, num_variants: int) -> list[PerturbedText]:
        """Convert characters to leetspeak equivalents."""
        results = []

        for _ in range(num_variants):
            perturbed = list(text.lower())
            changes = []

            # Find convertible characters
            convertible = [
                (i, c) for i, c in enumerate(text.lower()) if c in LEETSPEAK_MAP
            ]

            if not convertible:
                continue

            # Convert 2-4 characters randomly
            num_conversions = min(random.randint(2, 4), len(convertible))
            to_convert = random.sample(convertible, num_conversions)

            for idx, char in to_convert:
                replacement = random.choice(LEETSPEAK_MAP[char])
                perturbed[idx] = replacement
                changes.append(f"'{char}'→'{replacement}'")

            perturbed_text = "".join(perturbed)
            if perturbed_text != text.lower():
                results.append(
                    self._create_result(
                        original=text,
                        perturbed=perturbed_text,
                        method="leetspeak",
                        changes=changes,
                    )
                )

        return results

    def _spacing_perturb(self, text: str, num_variants: int) -> list[PerturbedText]:
        """Insert spaces within words to break pattern matching."""
        results = []
        words = text.split()

        for _ in range(num_variants):
            if not words:
                continue

            # Pick a word to add spacing to
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx]

            if len(word) < 3:
                continue

            # Insert spaces
            spaced_word = self._add_internal_spaces(word)

            if spaced_word != word:
                perturbed_words = words.copy()
                perturbed_words[word_idx] = spaced_word
                perturbed = " ".join(perturbed_words)

                results.append(
                    self._create_result(
                        original=text,
                        perturbed=perturbed,
                        method="word_spacing",
                        changes=[f"'{word}'→'{spaced_word}'"],
                    )
                )

        return results

    def _add_internal_spaces(self, word: str) -> str:
        """Add spaces within a word."""
        if len(word) <= 2:
            return word

        # Different spacing patterns
        patterns = [
            # Space after each character: "hack" -> "h a c k"
            lambda w: " ".join(list(w)),
            # Space in middle: "hack" -> "ha ck"
            lambda w: w[: len(w) // 2] + " " + w[len(w) // 2 :],
            # Alternating spaces: "hack" -> "h ack"
            lambda w: w[0] + " " + w[1:],
        ]

        return random.choice(patterns)(word)

    def _synonym_perturb(self, text: str, num_variants: int) -> list[PerturbedText]:
        """Replace words with synonyms."""
        results = []
        words = text.lower().split()

        # Find words that have synonyms
        replaceable = [
            (i, w) for i, w in enumerate(words) if w.strip(".,!?") in SYNONYMS
        ]

        if not replaceable:
            return results

        for _ in range(num_variants):
            perturbed_words = text.split()
            changes = []

            # Replace 1-2 words
            num_replacements = min(random.randint(1, 2), len(replaceable))
            to_replace = random.sample(replaceable, num_replacements)

            for idx, word in to_replace:
                # Strip punctuation for lookup
                clean_word = word.strip(".,!?")
                if clean_word in SYNONYMS:
                    synonym = random.choice(SYNONYMS[clean_word])
                    # Preserve punctuation
                    original_word = perturbed_words[idx]
                    punct_match = re.search(r"[.,!?]+$", original_word)
                    if punct_match:
                        synonym += punct_match.group()

                    perturbed_words[idx] = synonym
                    changes.append(f"'{clean_word}'→'{synonym}'")

            perturbed = " ".join(perturbed_words)
            if perturbed.lower() != text.lower() and changes:
                results.append(
                    self._create_result(
                        original=text,
                        perturbed=perturbed,
                        method="synonym",
                        changes=changes,
                    )
                )

        return results
