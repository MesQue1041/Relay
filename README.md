# Relay: a real-time voice assistant with a decomposed latency budget

A voice-in, voice-out assistant you talk to like a person: say something,
pause, and it answers out loud. Under the hood it's five stages — mic
capture, voice activity detection, speech-to-text, an LLM with tool-calling,
and text-to-speech — each one timed independently, so instead of one vague
"it feels slow" you get a p50/p95 breakdown of exactly which stage is
costing you time. When a stage is slow or fails outright, the assistant
degrades instead of hanging: it falls back to a shorter answer, tells you
out loud that something went wrong, or just keeps going with what it has.

## Quick start

    pip install -r requirements.txt
    cp .env.example .env          # add your GROQ_API_KEY
    python -m src.main            # talk to it
    python -m evals.run_evals     # score the agent on 13 fixed utterances
    pytest tests/                 # offline tests, no mic or API calls needed

Get a free Groq key from console.groq.com. On Windows, `zoneinfo` has no
system timezone database to read from, so `tzdata` is in requirements.txt
specifically for that — without it, asking for the time in a named city
fails outright.

## How it works

    mic ─► VAD (WebRTC + adaptive noise floor) ──speech detected──► buffer
                                                        │ silence threshold hit
                                                        ▼
                                                  STT (faster-whisper, local)
                                                        │ transcript
                                                        ▼
                                                  guardrail (empty / oversized /
                                                  punctuation-only rejection)
                                                        │ allowed
                                                        ▼
                                  ┌─► LLM (streaming, tool_choice="auto") ──text──► sentence-
                                  │       │ tool calls                            chunked TTS
                                  │       ▼                                       ──► speaker
                                  │   get_current_time / get_weather / web_search
                                  └────── tool results appended, loop (max 3 rounds)

| File | Job |
|---|---|
| audio/capture.py | Mic input stream; soft-limits gain instead of hard-clipping |
| audio/vad.py | Speech/silence detection: WebRTC VAD gated by an adaptive RMS noise floor |
| audio/playback.py | Persistent output stream; decodes and plays edge-tts's mp3 chunks |
| stt.py | faster-whisper wrapper, warmed up once at startup |
| agent.py | The tool-calling loop: streaming, tool-call reassembly, round cap |
| tools.py | get_current_time, get_weather, web_search, and their JSON schemas |
| guardrail.py | Rejects empty, oversized, or punctuation-only transcripts before they reach the LLM |
| fallback.py | Thread-based timeout wrapper used around TTS synthesis+playback |
| latency.py | Per-stage timer, p50/p95 summary printed at shutdown |
| control.py | Shared stop_event so Ctrl+C is noticed mid-turn instead of only between turns |
| main.py | Wires all of the above into the live mic-to-speaker loop |
| evals/ | 13 fixed utterances scoring tool routing, response validity, and latency |

## Design decisions

**Tools are decided by the model, not by keyword matching.** An earlier
version forced a specific tool via `tool_choice` whenever a regex matched
phrases like "what time is it" — which meant the eval's "routing accuracy"
was really just testing the regex. It's `tool_choice="auto"` throughout now,
so routing failures actually mean something.

**The tool loop is capped, and the cap doesn't fall through to another LLM
call.** After `MAX_TOOL_ROUNDS` rounds, Relay stops asking the model
entirely and speaks the last tool's own result text. The alternative — one
more LLM call with `tool_choice="none"` to force a final answer — sounds
safer but isn't: some models will still attempt a tool call under "none,"
and Groq hard-rejects that combination rather than ignoring it. Using the
tool's own output sidesteps the failure mode
