import json
import time
from dataclasses import dataclass
from groq import Groq
from src.config import GROQ_API_KEY, LLM_MODEL, LLM_TIMEOUT_SECONDS, MAX_TOOL_ROUNDS
from src.tools import TOOLS, AVAILABLE_FUNCTIONS

_client = Groq(api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT_SECONDS)

SYSTEM_PROMPT = (
    "You are Relay, a concise voice assistant. "
    "Keep responses short, usually 1-2 sentences, because your responses "
    "will be spoken aloud. "
    "If the user explicitly asks you to search the web, you MUST use web_search. "
    "If the user explicitly asks for weather, you MUST use get_weather. "
    "If the user explicitly asks for the current time or date, you MUST use "
    "get_current_time — pass the place they named as the city argument (e.g. "
    "'Malaysia', 'Tokyo'); if they didn't name a place, call it with no arguments. "
    "For general questions that do not need a tool, answer directly."
)


@dataclass
class AgentResult:
    text: str
    tools_used: list[str]
    iterations: int
    duration: float


def _role_of(message):
    return message.get("role") if isinstance(message, dict) else getattr(message, "role", None)


def _content_of(message):
    return message.get("content") if isinstance(message, dict) else getattr(message, "content", None)


def _last_tool_result_text(messages) -> str:
    for message in reversed(messages):
        if _role_of(message) == "tool":
            content = _content_of(message)
            if content:
                return content
    return (
        "I found some information but had trouble putting it into words — "
        "please try asking again."
    )


def _execute_tool_calls(messages, tool_calls, tools_used):
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        tools_used.append(function_name)
        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            function_args = {}
        function = AVAILABLE_FUNCTIONS.get(function_name)
        if function is None:
            result = f"Error: unknown tool '{function_name}'."
        else:
            try:
                result = function(**function_args)
            except Exception as exc:
                result = f"Tool '{function_name}' failed: {exc}"
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": str(result),
        })


def run_agent_with_trace(user_text: str) -> AgentResult:
    start = time.perf_counter()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text.strip()},
    ]
    tools_used: list[str] = []
    rounds = 0

    while True:
        response = _client.chat.completions.create(
            model=LLM_MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
        )
        choice = response.choices[0].message

        if not choice.tool_calls:
            messages.append(choice)
            return AgentResult(
                text=(choice.content or "").strip(),
                tools_used=tools_used,
                iterations=rounds,
                duration=time.perf_counter() - start,
            )

        rounds += 1
        messages.append(choice)
        _execute_tool_calls(messages, choice.tool_calls, tools_used)

        if rounds >= MAX_TOOL_ROUNDS:
            return AgentResult(
                text=_last_tool_result_text(messages),
                tools_used=tools_used,
                iterations=rounds,
                duration=time.perf_counter() - start,
            )


def run_agent(user_text: str) -> str:
    return run_agent_with_trace(user_text).text


def stream_agent(user_text: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text.strip()},
    ]
    tools_used: list[str] = []
    rounds = 0

    while True:
        stream = _client.chat.completions.create(
            model=LLM_MODEL, messages=messages, tools=TOOLS,
            tool_choice="auto", stream=True,
        )

        collected_content = ""
        tool_call_chunks = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                collected_content += delta.content
                yield delta.content

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    entry = tool_call_chunks.setdefault(
                        tc_delta.index, {"id": None, "name": "", "arguments": ""}
                    )
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        entry["name"] += tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        entry["arguments"] += tc_delta.function.arguments

        if not tool_call_chunks:
            return 

        rounds += 1
        messages.append({
            "role": "assistant",
            "content": collected_content or None,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
                for call in tool_call_chunks.values()
            ],
        })

        for call in tool_call_chunks.values():
            tools_used.append(call["name"])
            try:
                args = json.loads(call["arguments"]) if call["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            function = AVAILABLE_FUNCTIONS.get(call["name"])
            if function is None:
                result = f"Error: unknown tool '{call['name']}'."
            else:
                try:
                    result = function(**args)
                except Exception as exc:
                    result = f"Tool '{call['name']}' failed: {exc}"
            print(f"\n[Agent] Tool used: {call['name']}")
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["name"],
                "content": str(result),
            })

        if rounds >= MAX_TOOL_ROUNDS:
            yield _last_tool_result_text(messages)
            return