"""Decoded task rewards using strict, non-executing standard-library parsers."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .data import PromptRecord, RewardFn


TEXT_REWARDS = {"text_exact_match", "numeric_answer", "json_match"}
_NUMBER = re.compile(
    r"[+-]?(?:(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]*)?|\.[0-9]+)"
    r"(?:[eE]([+-]?[0-9]+))?"
)


def _final_answer(text: str, *, numeric: bool = False) -> str:
    """Extract a terminal answer tag; numeric answers also allow boxes or ####.

    Markers must end the response. Unmarked prose is never searched for numbers.
    The last complete answer tag is accepted, but nested/unclosed tags are not.
    """
    text = text.strip()
    if text.endswith("</answer>"):
        start = text.rfind("<answer>")
        prefix = text[:start]
        if start >= 0 and prefix.rfind("<answer>") <= prefix.rfind("</answer>"):
            text = text[start + len("<answer>"):-len("</answer>")].strip()
    if not numeric:
        return text
    start = text.rfind("\\boxed{")
    if start >= 0:
        content_start = start + len("\\boxed{")
        depth = 1
        for index in range(content_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    if index == len(text) - 1:
                        return text[content_start:index].strip()
                    break
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    return text


def _text_answer(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("expected a nonempty text answer")
    answer = " ".join(unicodedata.normalize("NFKC", _final_answer(value)).split())
    if not answer:
        raise ValueError("expected a nonempty text answer")
    return answer


def _decimal_fraction(text: str) -> Fraction:
    match = _NUMBER.fullmatch(text.strip())
    if match is None:
        raise ValueError("expected a finite numeric scalar")
    exponent = match.group(1)
    if exponent is not None and abs(int(exponent)) > 1000:
        raise ValueError("numeric exponent magnitude exceeds 1000")
    return Fraction(Decimal(text.strip().replace(",", "")))


def _numeric_answer(value: Any) -> Fraction:
    """Compare exact scalars, with no tolerance or algebraic simplification.

    Signed decimal/scientific notation, grouped thousands, a/b and \\frac{a}{b}
    are accepted. A terminal percent divides by 100: 1/2 == 0.5 == 50%.
    Reference strings and predictions use identical final-answer extraction.
    Scalars are limited to 512 characters and decimal exponents to +/-1000.
    """
    if type(value) not in (str, int, float) or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError("expected a finite numeric answer")
    text = _final_answer(str(value), numeric=True)
    if len(text) > 512:
        raise ValueError("numeric answer exceeds 512 characters")
    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()
    latex_fraction = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
    parts = list(latex_fraction.groups()) if latex_fraction else text.split("/")
    if len(parts) not in (1, 2):
        raise ValueError("expected a single scalar or fraction")
    result = _decimal_fraction(parts[0])
    if len(parts) == 2:
        denominator = _decimal_fraction(parts[1])
        if denominator == 0:
            raise ValueError("numeric answer has a zero denominator")
        result /= denominator
    return result / 100 if percent else result


def _json_value(value: Any, *, decoded: bool = False) -> Any:
    """Tag JSON types so booleans cannot compare equal to numeric 0 or 1."""
    if value is None or type(value) in (bool, str):
        return (type(value).__name__, value)
    if type(value) in (int, float) or (decoded and type(value) is Decimal):
        number = Decimal(str(value))
        if not number.is_finite():
            raise ValueError("JSON numbers must be finite")
        return ("number", number)
    if isinstance(value, list):
        return ("array", [_json_value(item, decoded=decoded) for item in value])
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return ("object", {key: _json_value(item, decoded=decoded) for key, item in value.items()})
    raise ValueError("expected a JSON value with string object keys")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _json_answer(text: str) -> Any:
    """Parse a whole JSON value or one complete json/unlabelled fenced block."""
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    value = json.loads(text, object_pairs_hook=_unique_object, parse_float=Decimal)
    return _json_value(value, decoded=True)


def build_text_reward(spec: str, tokenizer: Any) -> RewardFn:
    """Build an outcome reward over decoded text and metadata['answer'].

    Text matching normalizes NFKC and whitespace, preserving case. Numeric
    matching uses exact rational values. JSON matching preserves array order
    and JSON types (integer/float values may match); object key order is ignored.
    Missing/invalid reference answers raise ValueError; invalid outputs score 0.
    Only terminal <answer> tags are extracted for text; numeric also supports
    balanced \\boxed{...} and #### suffixes. JSON requires complete JSON input.
    """
    parsers = {
        "text_exact_match": (_text_answer, _text_answer),
        "numeric_answer": (_numeric_answer, _numeric_answer),
        "json_match": (_json_value, _json_answer),
    }
    if spec not in parsers:
        raise ValueError(f"unknown text reward {spec!r}")
    if not callable(getattr(tokenizer, "decode", None)):
        raise ValueError(f"reward {spec!r} requires a tokenizer with decode()")
    reference_parser, prediction_parser = parsers[spec]

    def reward(record: PromptRecord, response_ids: list[int]) -> float:
        try:
            expected = reference_parser(record.metadata["answer"])
        except (KeyError, ValueError, ArithmeticError, RecursionError) as exc:
            raise ValueError(f"reward {spec!r} requires valid metadata['answer']: {exc}") from exc
        try:
            text = tokenizer.decode(response_ids, skip_special_tokens=True)
        except ValueError:
            return 0.0
        if not isinstance(text, str):
            raise ValueError("tokenizer.decode() must return text")
        try:
            actual = prediction_parser(text)
        except (ValueError, ArithmeticError, RecursionError):
            return 0.0
        return float(actual == expected)

    return reward
