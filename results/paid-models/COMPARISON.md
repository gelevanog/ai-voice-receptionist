# Paid models vs the free baseline

Run on 2026-10-12 on branch `eval/paid-models`. Only the receptionist's LLM changes. Speech stays local and the
same as the baseline: Silero VAD, faster-whisper `base.en` int8, Kokoro-82M for Callie's voice, Piper for the
callers. No paid speech APIs, no phone calls, no Twilio. Every model call went through the local budget proxy
(an OpenAI-compatible endpoint). The repo used it as `CALLIE_LLM_PROVIDER=openai` with the new explicit opt-in
`CALLIE_ALLOW_PAID_MODELS=1`, and `CALLIE_LLM_SEND_TEMPERATURE=false` for Claude, because the gateway rejects
`temperature`.

## What was run, and what is not comparable

| Run | Mode | Agent model | Scenarios | Files |
|---|---|---|---|---|
| **Baseline** (already in the repo) | end to end through audio | `inclusionai/ling-3.0-flash-sante:free` (+ `nemotron-3-super:free` fallback) | 23 core | `results/e2e.json` |
| Haiku, text ×3 | text-level | `claude-haiku-5-5` | 22 × 3 runs | `text-haiku{,-r2,-r3}.json` |
| Opus, text ×1 | text-level | `claude-opus-5-5` | 22 | `text-opus.json` |
| Qwen, text ×2 | text-level | `qwen3.7-plus` (temperature 0.3, like the baseline) | 22 × 2 runs | `text-qwen{,-r2}.json` |
| Haiku, audio | end to end through audio | `claude-haiku-5-5` | all 23 core | `e2e-haiku.json` |
| Opus / Qwen, audio | end to end through audio | `claude-opus-5-5` / `qwen3.7-plus` | the 7 the baseline failed | `e2e-opus-hard.json`, `e2e-qwen-hard.json` |
| Opus, audio, `max_tokens` 1024 | end to end through audio | `claude-opus-5-5` | 2 (sensitivity check, see below) | `e2e-opus-hard-maxtok1024.json` |

Every run has a call ledger next to it (`*.calls.jsonl`: per request status, latency, time to first token and
tokens). `summary.json` has the pooled numbers and the per-scenario matrix that the tables below come from.

**Confounds. Read these before the numbers.**

- **Caller model.** The baseline caller was `liquid/lfm-2.5-2.6b:free`. OpenRouter wasn't reachable in this run,
  so every paid run used **`claude-haiku-5-5` as the simulated caller**, the same caller for every agent model. A
  stronger caller gives clearer lines, but it also played the "indecisive" persona too well (see `change_mind_day`).
- **Code version.** The baseline was measured at commit `afa3420`. The paid runs used the current code, which has
  the post-run fixes listed in the README: the reschedule-booking guard, the digit-count re-ask for messages and a
  final model step without tools. Where it matters, I checked from the tool logs whether those guards fired (they
  did not, see the observations).
- **Text-level mode** (`callie eval text`, new) sends the caller's text straight to the agent, so speech
  recognition is perfect by construction. `silent_caller` is left out because rules handle it with no model call,
  which leaves 22 scenarios. The barge-in and backchannel checks are dropped, and an interruption becomes the
  caller's next turn. There is **no free-model text-level run**, because the free models couldn't be reached. For
  reference, the baseline audio run re-graded on text-mode criteria is **16/22**.
- **Hard subset.** The 7 scenarios were picked *because the baseline failed them*, so the baseline is 0/7 by
  construction and some regression to the mean is expected. Each is one run of one call, which is noisy.

## Main table

| Metric | Baseline (free, audio) | Haiku 5.5 | Opus 5.5 | Qwen 3.7 Plus |
|---|---|---|---|---|
| **Task success, text-level (22 scenarios)** | n/a (audio run re-graded: 16/22) | **62/66 (94%)**, runs 20, 21, 21 | **21/22** (1 run) | **44/44 (100%)**, runs 22, 22 |
| **Task success, audio, all 23 core** | **16/23 (70%)** | **21/23 (91%)** | not run | not run |
| Audio, hard subset (7 baseline failures) | 0/7 (by selection) | 5/7 (from the full run) | 4/7 (2/2 of its failures pass again with `max_tokens` 1024) | 3/7 |
| Calendar changes made with a read-back + a yes | 8/8 | text 30/30, audio 10/10 | text 10/10, audio 3/3 | text 22/22, audio 3/3 |
| `confirmed=true` attempts refused by the gate | 0 | 0 | 0 | text 0, audio **3** |
| Agent claims of a change that did not happen | 0 | 0 | 0 | text 0, audio **1** |
| Bookings on a slot no tool offered | 0 | 0 | 0 | 0 |
| Clock times the model said on its own | 0 | 0 | 0 | text 0, audio **1** |
| Escalations transferred when expected / unexpected transfers | 3/3, 0 | text 9/9, **1**; audio 3/3, 0 | text 3/3, 0; audio-hard **1** | text 6/6, 0; audio-hard 0 |
| Tool calls that returned an error | 0 / 76 | text 1 / 215, audio 0 / 71 | text 0 / 70, audio-hard **2 / 32** | text 0 / 145, audio-hard 0 / 34 |
| Advice sentences replaced by the safety filter | 0 | text 3, audio 1 | 0 | text 3, audio-hard 0 |
| Model time to first token, LLM turns, p50 / p95 | 3.14 s / 6.34 s | text **1.85 / 3.07 s**; audio 1.96 / 7.95 s | text 2.45 / 3.67 s; audio-hard 2.87 / 5.31 s | text 2.25 / 17.2 s*; audio-hard 2.26 / 4.02 s |
| Voice to voice, LLM turns, p50 / p95 (audio) | 4.71 s / 9.43 s | **3.96 s** / 9.29 s | 4.35 / 7.76 s (hard subset) | 4.32 / 5.80 s (hard subset) |
| LLM turns that needed the "One moment." filler (1.8 s) | 96% | text 60%, audio 67% | text 90% | text 98% |
| Agent tokens per text scenario (mean) | not tracked | ~10.8k | ~11.7k | ~7.5k |
| Real requests (agent + caller) | 389 (whole project) | 859 Haiku requests across all runs | 145 | 191 |
| **Cost** | $0 | **≈ $0.16** (all Haiku use, agent and caller: 1.32M in / 51k out) | **≈ $2.13** (450k in / 16k out) | **435,918 Qwen tokens** of 500k |

\* Qwen's text p95 comes from one slow spell: seven consecutive requests of about 18 s within two minutes. The
same model's p95 in the audio subset was 4.0 s. Tails in every paid run are mostly gateway incidents (10 × `403
Request not allowed` retried, one 30 s stall), not the models' own speed.

Spend: the proxy reports **$2.52** of Anthropic spend and **435,918** Qwen tokens for this project. The per-model
dollar figures above come from the ledgers' token counts at the brief's prices ($2.29 in total). The ~$0.24 gap
is on the proxy side, where I can't see it per model.

### Per scenario (passes / runs)

| Scenario | Baseline audio | Haiku text | Opus text | Qwen text | Haiku audio | Opus audio | Qwen audio |
|---|---|---|---|---|---|---|---|
| book_basic | P | 3/3 | 1/1 | 2/2 | 1/1 | | |
| book_specific_time | **F** (STT name/phone) | 3/3 | 1/1 | 2/2 | 1/1 | 0/1 (truncated tool call) | 0/1 (correction loop) |
| book_relative_date | P | 3/3 | 1/1 | 2/2 | 1/1 | | |
| change_mind_day | **F** (agent) | **1/3** (caller turn cap 2×) | **0/1** (caller turn cap) | 2/2 | 0/1 (caller turn cap) | 1/1 | 0/1 (wrong Wednesday) |
| decline_after_options | P | 3/3 | 1/1 | 2/2 | 1/1 | | |
| reschedule_basic, _caller_id | P, P | 6/6 | 2/2 | 4/4 | 2/2 | | |
| reschedule_interrupt | **F** (agent: wrong tool) | 3/3 | 1/1 | 2/2 | 1/1 | 0/1 (truncated tool call) | 0/1 (caller confused) |
| reschedule_noisy | **F** (STT, noise) | 3/3 | 1/1 | 2/2 | 1/1 | 1/1 | 1/1 |
| cancel_basic | P | 3/3 | 1/1 | 2/2 | 1/1 | | |
| faq_* (4) | 4/4 | 12/12 | 4/4 | 8/8 | 4/4 | | |
| after_hours_person | **F** (STT digits) | 3/3 | 1/1 | 2/2 | 1/1 | 1/1 | 1/1 |
| emergency, angry, human_request | 3/3 | 9/9 | 3/3 | 6/6 | 3/3 | | |
| medical_advice | **F** (STT phone) | 2/3 | 1/1 | 2/2 | 1/1 | 1/1 | 1/1 |
| wrong_number, non_native_speaker | P, P | 6/6 | 2/2 | 4/4 | 2/2 | | |
| silent_caller | P | (not in text mode) | | | 1/1 | | |
| backchannel_listener | **F** (STT) | 2/3 (truncated tool call) | 1/1 | 2/2 | 0/1 (STT) | 0/1 (STT) | 0/1 (STT) |

## Observations

- **Free was already enough for most of the call types.** FAQ, emergency, angry-caller and person requests,
  cancel, wrong number and simple bookings pass for every model in every mode, as they did for the free baseline.
  None of the code-level safety numbers got worse with any model on the 66 + 22 + 44 text calls and the 23-call
  Haiku audio run: every calendar change had a read-back and a yes, no slot was booked that wasn't offered, and
  no clock time was invented. The paid models only make a difference on the multi-step flows.
- **The audio score went from 16/23 to 21/23 with Haiku, but this one run can't credit that to the model.** Four
  of the five newly passing scenarios were blamed on speech recognition in the baseline. This time Whisper heard
  "Okafor" correctly, and the noisy-line caller was found on the second lookup. The code guards added after the
  baseline did not fire (no `phone_unclear` re-ask, no refused booking during a reschedule), so the fixes came
  from this run's model and caller plus speech-recognition luck. The clearest case of the model itself helping is
  `reschedule_interrupt`. The free model called `book_appointment` there; Haiku and Opus called
  `reschedule_appointment` in every text run and in Haiku's audio run.
- **Speech recognition is still the ceiling, whatever the model.** `backchannel_listener` fails in audio for all
  four agent models: Whisper turns "Mm-hmm" into "And then, hum.". In the hard subset, Opus and Qwen both lost
  `book_specific_time` in long name-spelling loops over audio. A stronger model doesn't fix a misheard name. The
  remedies in the README (caller ID, keypad entry, spelling) still apply.
- **A new failure mode with Claude through this gateway: tool calls cut off at `max_tokens`.** The agent's
  400-token cap was sized for the free models, whose largest answer in the baseline was 352 tokens. Claude's
  answers regularly use 100–400 tokens for a short tool call, which suggests hidden reasoning counts toward the
  cap; Qwen's largest was 124. Three times (Opus 2 of 7 audio calls, Haiku 1 of 66 text calls) the tool call was
  cut mid-JSON. The agent then put the invalid arguments back into the conversation, the gateway refused every
  later request (`400 Failed to parse JSON`), and after two errors the agent transferred the caller. Rerunning
  Opus's two failures with `CALLIE_LLM_MAX_TOKENS=1024` passed both (n=2). Before using these models in
  production, the cap should be raised and invalid tool arguments should be replaced before they go back into
  the history. Neither change was made here, because it would change the agent compared with the baseline.
- **Qwen 3.7 Plus has the best text score (44/44) and the fewest tokens, but in audio it was the least
  careful.** In 7 hard audio calls it tried to confirm changed details three times (the gate refused all three),
  told a caller "You're all set" when nothing had been booked, wrote its own availability sentence ("Thursday at
  2:30 PM is open") and booked this week's Wednesday when the caller had moved from "next Monday" to "Wednesday
  afternoon". The code guards kept the calendar correct, but the false claim reached the caller.
- **Latency.** Haiku has the fastest model step: 1.85 s median to first token, against 3.14 s for the free
  baseline. Its median voice-to-voice time in audio is 3.96 s against 4.71 s. Opus (about 2.5 s) and Qwen (about
  2.25 s) also beat the free tier at the median. Even so, the 1.8 s filler still plays in 60–98% of model turns.
  To answer without it, the threshold or the prompt size (~2.7k tokens per request) would have to change. On
  `change_mind_day`, the Haiku caller kept changing its mind until the six-turn cap, which cost Haiku 2 of 3 runs
  and Opus its only text run. That is a weakness of the simulated caller, not of the agent.

## How to reproduce

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1 OPENAI_API_KEY=proj-callie   # the budget proxy
export CALLIE_ALLOW_PAID_MODELS=1 CALLIE_NOW=2026-10-06T09:30 CALLIE_LLM_PROVIDER=openai
export CALLIE_LLM_MODEL=claude-haiku-5-5 CALLIE_LLM_SEND_TEMPERATURE=false   # true for qwen3.7-plus
export CALLIE_LLM_LEDGER=results/paid-models/text-haiku.calls.jsonl CALLIE_LLM_MAX_CALLS=3000
# text-level, 22 scenarios
uv run callie eval text --caller-provider openai --caller-model claude-haiku-5-5 --no-caller-temperature \
  --out results/paid-models/text-haiku.json
# end to end through audio (needs `make models`), same caller
CALLIE_STT_PROVIDER=whisper CALLIE_TTS_PROVIDER=kokoro CALLIE_VAD_PROVIDER=silero \
  uv run callie eval run --caller llm --caller-provider openai --caller-model claude-haiku-5-5 \
  --no-caller-temperature --no-cache --out-dir results/paid-models --tag e2e-haiku
```

Without `CALLIE_ALLOW_PAID_MODELS=1`, both commands refuse any agent or caller model that isn't the fake model or
a `:free` OpenRouter id, and they refuse before any request is sent.
