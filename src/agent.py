import json
from groq import Groq
from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
)
from src.tools import (
    TOOLS,
    AVAILABLE_FUNCTIONS,
)

_client = Groq(
    api_key=GROQ_API_KEY,
    timeout=LLM_TIMEOUT_SECONDS,
)

SYSTEM_PROMPT = (
    "You are Relay, a concise voice assistant. "
    "Keep responses short, usually 1-2 sentences, because "
    "your responses will eventually be spoken aloud. "
    "Use the available tools when current information is needed, "
    "especially for time, weather, or web facts. "
    "For general questions that do not need a tool, answer directly."
)

def run_agent(user_text: str) -> str:
    if not user_text or not user_text.strip():
        return ""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_text.strip(),
        },
    ]

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    choice = response.choices[0].message

    if not choice.tool_calls:
        return (choice.content or "").strip()
 
    messages.append(choice)

    for tool_call in choice.tool_calls:
        function_name = tool_call.function.name

        try:
            function_args = json.loads(
                tool_call.function.arguments
            )
        except json.JSONDecodeError:
            function_args = {}

        function = AVAILABLE_FUNCTIONS.get(function_name)

        if function is None:
            result = (
                f"Error: unknown tool '{function_name}'."
            )

        else:
            try:
                result = function(**function_args)
            except Exception as exc:
                result = (
                    f"Tool '{function_name}' failed: {exc}"
                )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(result),
            }
        )

    followup = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
    )

    return (
        followup.choices[0].message.content or ""
    ).strip()