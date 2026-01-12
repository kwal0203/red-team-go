"""Character-level text perturbations.

Includes typos, unicode homoglyphs, and invisible character insertions.
"""

import random

from services.evaluation.benchmarks.robustness.src.perturbations.base import (
    BasePerturbation,
    PerturbedText,
)

# Unicode homoglyphs - characters that look similar to ASCII
HOMOGLYPHS = {
    "a": ["а", "ɑ", "α"],  # Cyrillic а, Latin alpha, Greek alpha
    "c": ["с", "ϲ"],  # Cyrillic с, Greek lunate sigma
    "e": ["е", "ε", "ё"],  # Cyrillic е, Greek epsilon
    "i": ["і", "ι", "ı"],  # Cyrillic і, Greek iota, Turkish dotless i
    "o": ["о", "ο", "σ"],  # Cyrillic о, Greek omicron
    "p": ["р", "ρ"],  # Cyrillic р, Greek rho
    "s": ["ѕ", "ꜱ"],  # Cyrillic ѕ, small capital S
    "x": ["х", "χ"],  # Cyrillic х, Greek chi
    "y": ["у", "γ"],  # Cyrillic у, Greek gamma
    "A": ["А", "Α"],  # Cyrillic А, Greek Alpha
    "B": ["В", "Β"],  # Cyrillic В, Greek Beta
    "C": ["С", "Ϲ"],  # Cyrillic С, Greek Sigma
    "E": ["Е", "Ε"],  # Cyrillic Е, Greek Epsilon
    "H": ["Н", "Η"],  # Cyrillic Н, Greek Eta
    "K": ["К", "Κ"],  # Cyrillic К, Greek Kappa
    "M": ["М", "Μ"],  # Cyrillic М, Greek Mu
    "O": ["О", "Ο"],  # Cyrillic О, Greek Omicron
    "P": ["Р", "Ρ"],  # Cyrillic Р, Greek Rho
    "T": ["Т", "Τ"],  # Cyrillic Т, Greek Tau
    "X": ["Х", "Χ"],  # Cyrillic Х, Greek Chi
}

# Invisible/zero-width characters
INVISIBLE_CHARS = [
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\ufeff",  # Zero-width no-break space
]

# Keyboard adjacent characters for typo simulation
ADJACENT_KEYS = {
    "a": "sqwz",
    "b": "vghn",
    "c": "xdfv",
    "d": "erfcxs",
    "e": "wrsdf",
    "f": "rtgvcd",
    "g": "tyhbvf",
    "h": "yujnbg",
    "i": "ujkol",
    "j": "uikmnh",
    "k": "iojlm",
    "l": "opk",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "edft",
    "s": "weadzx",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qeas",
    "x": "zsdc",
    "y": "tghu",
    "z": "asx",
}


class CharacterPerturbation(BasePerturbation):
    """Character-level perturbations including typos, homoglyphs, and invisible chars."""

    name = "character"
    category = "character"

    def perturb(self, text: str, num_variants: int = 5) -> list[PerturbedText]:
        """Generate character-level perturbations.

        Args:
            text: Original text to perturb.
            num_variants: Number of variants per method.

        Returns:
            List of perturbed text variants.
        """
        results: list[PerturbedText] = []

        # Generate homoglyph variants
        results.extend(self._homoglyph_perturb(text, num_variants))

        # Generate typo variants
        results.extend(self._typo_perturb(text, num_variants))

        # Generate invisible char variants
        results.extend(self._invisible_char_perturb(text, num_variants))

        return results

    def _homoglyph_perturb(self, text: str, num_variants: int) -> list[PerturbedText]:
        """Replace characters with visually similar Unicode homoglyphs."""
        results = []

        for _ in range(num_variants):
            perturbed = list(text)
            changes = []

            # Find replaceable characters
            replaceable = [(i, c) for i, c in enumerate(text) if c in HOMOGLYPHS]

            if not replaceable:
                continue

            # Replace 1-3 characters randomly
            num_replacements = min(random.randint(1, 3), len(replaceable))
            to_replace = random.sample(replaceable, num_replacements)

            for idx, char in to_replace:
                replacement = random.choice(HOMOGLYPHS[char])
                perturbed[idx] = replacement
                changes.append(f"'{char}'→'{replacement}' at pos {idx}")

            perturbed_text = "".join(perturbed)
            if perturbed_text != text:  # Only add if actually changed
                results.append(
                    self._create_result(
                        original=text,
                        perturbed=perturbed_text,
                        method="unicode_homoglyph",
                        changes=changes,
                    )
                )

        return results

    def _typo_perturb(self, text: str, num_variants: int) -> list[PerturbedText]:
        """Simulate keyboard typos by replacing with adjacent keys."""
        results = []

        for _ in range(num_variants):
            perturbed = list(text)
            changes = []

            # Find replaceable characters (letters only)
            replaceable = [
                (i, c.lower()) for i, c in enumerate(text) if c.lower() in ADJACENT_KEYS
            ]

            if not replaceable:
                continue

            # Replace 1-2 characters
            num_replacements = min(random.randint(1, 2), len(replaceable))
            to_replace = random.sample(replaceable, num_replacements)

            for idx, char in to_replace:
                replacement = random.choice(ADJACENT_KEYS[char])
                # Preserve case
                if text[idx].isupper():
                    replacement = replacement.upper()
                perturbed[idx] = replacement
                changes.append(f"typo '{text[idx]}'→'{replacement}' at pos {idx}")

            perturbed_text = "".join(perturbed)
            if perturbed_text != text:
                results.append(
                    self._create_result(
                        original=text,
                        perturbed=perturbed_text,
                        method="typo",
                        changes=changes,
                    )
                )

        return results

    def _invisible_char_perturb(
        self, text: str, num_variants: int
    ) -> list[PerturbedText]:
        """Insert invisible/zero-width characters."""
        results = []
        words = text.split()
        is_single_word = len(words) < 2

        if is_single_word:
            # For single word, insert within the word
            # Skip if text too short (need at least 3 chars to insert in middle)
            if len(text) > 2:
                for _ in range(num_variants):
                    pos = random.randint(1, len(text) - 1)
                    invisible = random.choice(INVISIBLE_CHARS)
                    perturbed = text[:pos] + invisible + text[pos:]
                    results.append(
                        self._create_result(
                            original=text,
                            perturbed=perturbed,
                            method="invisible_char",
                            changes=[f"inserted zero-width char at pos {pos}"],
                        )
                    )
        else:
            # Insert between words or within a word
            for _ in range(num_variants):
                perturbed_words = words.copy()
                word_idx = random.randint(0, len(words) - 1)
                word = perturbed_words[word_idx]

                # Skip words too short to insert in middle
                if len(word) > 2:
                    pos = random.randint(1, len(word) - 1)
                    invisible = random.choice(INVISIBLE_CHARS)
                    perturbed_words[word_idx] = word[:pos] + invisible + word[pos:]
                    perturbed = " ".join(perturbed_words)
                    results.append(
                        self._create_result(
                            original=text,
                            perturbed=perturbed,
                            method="invisible_char",
                            changes=[f"inserted zero-width char in word '{word}'"],
                        )
                    )

        return results
