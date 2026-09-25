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
tool's own output sidesteps the failure mode completely.

**TTS timeout scales with sentence length.** A flat timeout either fires
false-positive on long sentences (a 6-second reply needs more than 5 seconds
to *speak*, let alone synthesize) or is too generous to catch a real hang on
short ones. The timeout is computed per sentence from an estimated speaking
rate plus fixed connection overhead.

**The output audio stream is opened once and reused.** Opening a new
`sd.OutputStream` per sentence meant renegotiating with the OS audio engine
every time, which showed up as real, audible dead air after every reply.
The input stream and Whisper model get the same treatment — warmed up once
at startup instead of paying that cost on the first real utterance.

**Mic gain is a soft limiter, not a hard clip.** A flat gain multiply
followed by clipping distorts the waveform at the ceiling, and that
distortion is exactly what degrades transcription quality. A tanh-based
soft knee compresses loud peaks instead of chopping them, while leaving
quiet audio untouched.

**Ctrl+C is a shared event, not a caught exception.** Blocking calls inside
audio playback and queue joins don't reliably surface `KeyboardInterrupt`
promptly. A SIGINT handler sets a `stop_event` that every loop polls, and a
second Ctrl+C forces an immediate exit rather than waiting on cleanup that
might itself be stuck.

**The guardrail is a cheap rule check, not a model call.** Scope-creep
protection here doesn't need an LLM: the actual risk in a voice pipeline is
garbled or empty transcripts reaching the agent, not off-topic misuse, so a
length/content check catches what matters without adding a second network
round-trip to every turn.

## A real run

Eval harness, 13 fixed utterances:

    [1/13] time_01                    Routing: PASS   Latency: 1585ms
    [3/13] weather_01                 Routing: PASS   Latency: 3191ms
    [6/13] search_01                  Routing: PASS   Latency: 2239ms
    [8/13] direct_01                  Routing: PASS   Latency: 679ms
    ...
    Tool routing:     13/13 (100.0%)
    Valid responses:  13/13 (100.0%)
    Agent latency p50: 1482ms
    Agent latency p95: 3051ms

Live session, per-stage latency summary at shutdown:

    stt             count=  7  p50=    780 ms  p95=    836 ms
    llm_ttft        count=  7  p50=   1544 ms  p95=   4692 ms
    tts_ttfa        count=  7  p50=   1092 ms  p95=   1266 ms
    tts             count=  7  p50=   7062 ms  p95=  13580 ms
    turn_total      count=  7  p50=  11092 ms  p95=  16081 ms

TTS is consistently the long pole, mostly network round-trip to edge-tts
rather than anything in the pipeline itself — `tts_ttfa` (time to *first*
audio byte) stays under 1.3s even when total synthesis time for a full
sentence stretches past 13s.

## Known limits

No memory across turns — each utterance is handled independently, so
follow-ups like "what about tomorrow" have no prior context to resolve
against. No wake word or push-to-talk, so the VAD will trigger on any
detected speech in the room, background conversation included. The eval
harness scores tool routing, response validity, and latency on text
input directly; it doesn't measure STT transcription accuracy, since
that would need real recorded audio per case rather than typed
utterances. Mic gain is a fixed multiplier tuned for one quiet
microphone and may need retuning on a different machine. There's no
offline TTS fallback — if edge-tts's endpoint is unreachable, Relay logs
the reply as text and moves on rather than speaking it another way. A
Ctrl+C mid-sentence can only be noticed between audio chunks that have
already arrived over the network, not during an in-flight read, so it
occasionally takes a moment longer than expected to actually stop.
