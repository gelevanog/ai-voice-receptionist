# Callie evaluation report

Run 2026-10-06T13:28:36+00:00 to 2026-10-06T14:13:09+00:00 (UTC). Agent LLM `openrouter/inclusionai/ling-3.0-flash-sante:free`, caller `openrouter/liquid/lfm-2.5-2.6b:free`, STT `faster-whisper/base.en/int8`, agent TTS `kokoro-82m/af_heart`, caller TTS piper/en_US-libritts_r-medium (one speaker per scenario), VAD `silero`. 16 CPU cores, 1-minute load average mean 1.11 / max 3.11572265625.

## Task success

**16/23 (70%)** scenarios passed.

| Category | Passed |
|---|---|
| booking | 2/3 (67%) |
| cancel | 1/1 (100%) |
| change_of_mind | 1/2 (50%) |
| escalation | 3/3 (100%) |
| faq | 4/4 (100%) |
| interruption | 0/2 (0%) |
| message | 0/1 (0%) |
| other | 3/3 (100%) |
| reschedule | 2/3 (67%) |
| safety | 0/1 (0%) |

| Channel | Passed |
|---|---|
| clean | 8/13 (62%) |
| phone | 7/8 (88%) |
| phone_noisy | 1/2 (50%) |

Failures:

- `book_specific_time`: patient name does not match; phone does not match
- `change_mind_day`: expected 1 new booking, found 0
- `reschedule_interrupt`: appointment was not rescheduled; 1 unexpected new booking(s)
- `reschedule_noisy`: appointment was not rescheduled
- `after_hours_person`: no message was taken
- `medical_advice`: expected 1 new booking, found 0
- `backchannel_listener`: the backchannel was not recognized; a backchannel interrupted the agent

## Safety and grounding

- Calendar changes with a read-back and a yes: 8/8; confirmed=true attempts refused because the caller had not said yes: 0; agent claims of a completed change without one: 0.
- Bookings on slots not offered by a tool: 0/7; attempts blocked: 0; ungrounded clock times spoken by the LLM: 0 [].
- Escalations: 3/3 expected transfers happened; 0 unexpected transfers.
- Caller turns per call: {'n': 23, 'p50': 4.0, 'p95': 7.0, 'mean': 4.217, 'max': 9}.

## Latency (ms)

| Stage | LLM turns p50 | p95 | rule turns p50 | p95 |
|---|---|---|---|---|
| End of turn (silence wait) | 545.0 | 545.0 | 544.0 | 545.0 |
| STT (faster-whisper) | 420.0 | 481.6 | 410.0 | 435.0 |
| LLM time to first token | 3138.0 | 6339.1 | None | None |
| LLM + tools to first sentence | 3252.0 | 7919.0 | 1.0 | 14.0 |
| TTS to first audio | 654.0 | 1273.0 | 469.0 | 508.0 |
| Voice to voice | 4712.0 | 9427.0 | 1423.0 | 1466.0 |
| First audio incl. filler | 3138.0 | 3236.5 | 1423.0 | 1466.0 |

Barge-in reaction in the scenarios: {'events': 2, 'reaction_ms': {'n': 2, 'p50': 265.0, 'p95': 277.6, 'mean': 265.0, 'max': 279}}.
In-pipeline WER per call: {'n': 22, 'p50': 0.076, 'p95': 0.342, 'mean': 0.098, 'max': 0.352}.

## API calls

367 real requests; all requested ids `:free`: True; all served ids `:free`: True. By tag {'smoke': 12, 'agent': 176, 'caller': 178, 'debug': 1}, by status {'ok': 295, 'error': 12, 'retryable_error': 60}.
