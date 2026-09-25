from dataclasses import dataclass
from src.config import GUARDRAIL_MAX_INPUT_CHARS

@dataclass
class GuardrailResult:
    allowed: bool
    text: str
    reason: str = ""


def validate_input(user_text: str) -> GuardrailResult:
    if not user_text:
        return GuardrailResult(
            allowed=False,
            text="",
            reason="empty input",
        )
    cleaned = user_text.strip()
    if not cleaned:
        return GuardrailResult(
            allowed=False,
            text="",
            reason="empty input",
        )
    if len(cleaned) > GUARDRAIL_MAX_INPUT_CHARS:
        return GuardrailResult(
            allowed=False,
            text=cleaned,
            reason=(
                f"input exceeded {GUARDRAIL_MAX_INPUT_CHARS} characters"
            ),
        )
    
    alphanumeric_count = sum(
        character.isalnum()
        for character in cleaned
    )

    if alphanumeric_count < 2:
        return GuardrailResult(
            allowed=False,
            text=cleaned,
            reason="transcript contains insufficient meaningful text",
        )

    return GuardrailResult(
        allowed=True,
        text=cleaned,
    )

def guardrail_response(reason: str) -> str:
    if "exceeded" in reason:
        return (
            "That was a little too long for one voice turn. "
            "Please try a shorter question."
        )

    return (
        "I couldn't make out a useful request. "
        "Please try saying that again."
    )