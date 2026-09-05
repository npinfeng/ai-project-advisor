"""Dependency-free multilingual text normalization for lexical retrieval."""

from __future__ import annotations

import re
import unicodedata


_TERM_PATTERN = re.compile(
    r"[a-z0-9]+(?:[._+#/-][a-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+",
    re.IGNORECASE,
)
_CJK_PATTERN = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


def normalize_text(text: str) -> str:
    """Normalize width/case while preserving technical punctuation."""
    return unicodedata.normalize("NFKC", str(text or "")).casefold()


def lexical_tokens(text: str) -> list[str]:
    """Tokenize Latin technical terms and Chinese without external dictionaries.

    Chinese runs use character bigrams plus the complete short phrase. Bigrams
    make unseen compound words searchable, while complete phrases reward exact
    matches. Single-character runs are retained for short product/API names.
    """
    tokens: list[str] = []
    normalized_text = unicodedata.normalize("NFKC", str(text or ""))
    for match in _TERM_PATTERN.finditer(normalized_text):
        raw_term = match.group(0)
        term = raw_term.casefold()
        if not _CJK_PATTERN.fullmatch(term):
            if len(term) > 1 or term.isdigit():
                tokens.append(term)
            # Keep an exact technical compound and its searchable components.
            # Splitting before case-folding also covers CamelCase identifiers.
            parts = re.split(r"[._+#/-]+|(?<=[a-z0-9])(?=[A-Z])", raw_term)
            if len(parts) > 1:
                tokens.extend(
                    part.casefold() for part in parts
                    if len(part) > 1 or part.isdigit()
                )
            continue
        if len(term) == 1:
            tokens.append(term)
            continue
        tokens.extend(term[index:index + 2] for index in range(len(term) - 1))
        if len(term) <= 12:
            tokens.append(term)
    return tokens


def lexical_overlap(query: str, text: str) -> float:
    """Return query-token coverage in 0..1, with a small exact-phrase bonus."""
    query_tokens = set(lexical_tokens(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(lexical_tokens(text))
    coverage = len(query_tokens & text_tokens) / len(query_tokens)
    normalized_query = normalize_text(query).strip()
    if normalized_query and normalized_query in normalize_text(text):
        coverage = min(1.0, coverage + 0.15)
    return coverage
