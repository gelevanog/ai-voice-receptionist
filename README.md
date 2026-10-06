# Callie: an AI voice receptionist that answers calls, books appointments and hands off to a human

**A real-time voice agent for small service businesses. It picks up phone and browser calls, answers questions from the business's own knowledge base, books, moves and cancels appointments in the real calendar after reading the details back, and transfers the caller to a person when it should. Speech runs locally on CPU; the language model is any OpenAI-compatible endpoint, Anthropic, or free OpenRouter models.**

[![CI](https://github.com/gelevanog/ai-voice-receptionist/actions/workflows/ci.yml/badge.svg)](https://github.com/gelevanog/ai-voice-receptionist/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSockets-009688?logo=fastapi&logoColor=white)
![Twilio](https://img.shields.io/badge/Twilio-Media%20Streams-F22F46?logo=twilio&logoColor=white)
![STT](https://img.shields.io/badge/STT-faster--whisper%20on%20CPU-FFD21E)
![TTS](https://img.shields.io/badge/TTS-Kokoro--82M%20on%20CPU-8A2BE2)
![mypy strict](https://img.shields.io/badge/mypy-strict-2a6db2)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

https://github.com/user-attachments/assets/4b9bc781-1140-4ad4-aa0f-0fbc8b5dc779

<sub>70-second walkthrough with voiceover and an excerpt of a real browser call. Can't play it? [Download the MP4](docs/demo.mp4).</sub>

![The live call page during a real browser call: transcript, tool calls and the per-turn latency waterfall](docs/screenshots/live-call.png)

<sub>A real browser call, placed in headless Chrome with a synthetic caller voice as the microphone: Silero VAD, faster-whisper `base.en` and Kokoro-82M on CPU, the free model `inclusionai/ling-3.0-flash-sante:free` deciding. Callie offers only the times the calendar returned, reads the booking back, and the plain "yes" is executed by a rule without another model call (turn #4, 1.27 s voice to voice). Names and numbers are masked in the live transcript. [Full page](docs/screenshots/live-call-full.png).</sub>

🎧 **Listen:** [a booking call](docs/demo-calls/booking-call.mp3) ([transcript](docs/demo-calls/booking-call.md)) and [a reschedule where the caller interrupts Callie](docs/demo-calls/interruption-reschedule-call.mp3) ([transcript](docs/demo-calls/interruption-reschedule-call.md)).

**Measured on 2026-10-06** on a 16-core CPU without a GPU (shared machine, 1-minute load average 1.1 during the run, max 3.1), with free OpenRouter models:

| | Result |
|---|---|
| Simulated calls handled correctly, end to end through audio (23 scenarios, LLM caller, synthetic voices) | **16 of 23** (70%): booking, reschedule, cancel, FAQ, escalation, wrong number, silence; the 7 failures are analyzed below |
| Calendar changes made with a read-back and a yes / bookings on a time no tool offered | **8 of 8** / **0** (and 0 clock times invented by the model) |
| Escalations: emergency, angry caller, request for a person | **3 of 3** transferred, **0** unexpected transfers |
| Voice-to-voice latency, turns that needed the LLM (p50 / p95) | **4.7 s / 9.4 s**, of which the free model's first token is 3.1 s / 6.3 s; first audio incl. a "One moment." filler **3.1 s** |
| Voice-to-voice latency, turns handled by rules (a "yes" to a read-back, escalations, goodbyes) | **1.4 s / 1.5 s** |
| Barge-in: reaction time (p50 / p95), interruptions handled, backchannels handled | **271 / 326 ms**; **20 of 20**; **18 of 20** ("mm-hmm" twice transcribed as "and then, hum") |
| Speech recognition WER (`base.en`), clean / phone line 8 kHz / phone line + noise | **11.2% / 11.7% / 20.0%** on 94 synthetic caller utterances |
| Real API calls for everything, all `:free` | **389** (0 paid; the free-only guard refuses anything else) |

**The honest verdict:** the parts that are code work as designed: in every scenario nothing was booked, moved or cancelled without a read-back and a clear yes, no time was offered that the calendar did not return, every escalation reached a person, and barge-in reacts in about a quarter of a second. What fails is the hard part of voice: speech recognition of names and phone numbers (five of the seven failures; a noisy phone line made one caller unidentifiable), and a free model that sometimes calls the wrong tool (it once booked a second appointment instead of moving one; a guard added after the run now refuses that). Latency is dominated by the free model's first token (about 3 s, up to 13 s when the free tier queues); speech recognition and synthesis on CPU add about one second. Details, every failure and every number's source are below.

## What problem it solves

A dental practice, a garage or a salon misses calls all the time: after hours, at lunch, while the front desk is with a patient. Most people who reach voicemail do not leave a message; they call the next business on the list. Every missed call is a lost booking.

Callie answers every call, day and night. It says up front that it is an AI assistant and that the call may be recorded, answers questions about hours, prices, insurance, parking and services from the business's own information, and books, reschedules or cancels appointments in the business's real calendar. It never invents an opening: times come only from the calendar, and nothing is booked until Callie has read the name, service, date and time back and the caller has said yes. When a call needs a person (an emergency, an upset caller, a request for the front desk, or a conversation that is going in circles) it transfers the call, or takes a message when nobody is in. Confirmations go out by text.

The demo business is **Brightside Dental**, a fictional clinic in New York time. Every name, number and appointment in this repository is made up.

## Features

- **Real-time voice pipeline** (Python, FastAPI, WebSockets, asyncio), streamed at every stage:
  - **voice activity detection** with Silero VAD (ONNX, 32 ms frames) and **endpointing** that ends the caller's turn after 550 ms of silence (configurable; the latency/turn-taking trade-off is measured below);
  - **speech to text** with faster-whisper `base.en` (int8, CPU), chosen by measuring `tiny.en`, `base.en` and `small.en` on this machine;
  - **LLM with tool calling**, streamed; text is cut into sentences as it arrives and the **first clause goes to TTS before the answer is complete**;
  - **text to speech** with Kokoro-82M (ONNX, CPU, Apache-2.0 voice), one sentence at a time;
  - **barge-in**: when the caller talks over Callie, playback pauses within ~250 ms of speech; a **backchannel** ("mm-hmm", "okay") resumes it, anything else drops the rest of the answer, cancels generation and keeps only the words the caller actually heard in the conversation history;
  - **turn-taking**: utterances spoken while Callie is still thinking are merged into one turn; a filler ("One moment.") plays when the model is slow; **silence prompts** ("Are you still there?") and a polite hang-up.
- **Telephony**
  - **Browser calls** (the demo): microphone in an AudioWorklet (16 kHz PCM16), playback with a 120 ms buffer that is cleared instantly on barge-in, live transcript, tool calls and a per-turn **latency waterfall**.
  - **Twilio Media Streams**: TwiML webhook with signature validation, bidirectional WebSocket with base64 G.711 μ-law at 8 kHz, click-free stateful resampling, `clear` on barge-in, `mark` messages, DTMF "0 for a person", transfer by updating the live call with `<Dial>` TwiML. Tested with recorded message fixtures; see the honest note under [Connect a Twilio number](#connect-a-twilio-number).
- **Agent and tools**: `check_availability`, `book_appointment`, `find_appointment`, `reschedule_appointment`, `cancel_appointment`, `answer_faq` (BM25 retrieval over the clinic's Markdown knowledge base), `take_message`, `transfer_to_human`, `send_confirmation_sms` (Twilio SMS; dry-run outbox without credentials), `end_call`.
- **Calendar** in SQLite (SQLAlchemy 2.0, Postgres-ready): business hours, lunch break, holidays, service durations, two resources (hygienist, dentist), conflicts per resource, minimum notice, booking horizon, demo seed data. Optional **Google Calendar connector** (Calendar API v3): busy times block slots, bookings are mirrored as events; tested against a mocked API.
- **Safety and reliability, enforced in code, not only in the prompt**
  - **Read-back and a yes before any change**: the first `book/reschedule/cancel` call returns a deterministic read-back; a confirmed call succeeds only if the details match and the caller's very next turn was a clear yes. A plain "yes" executes the action without another model call.
  - **Availability only from tools**: booking accepts only slot ids that `check_availability` offered in the same call; the calendar re-checks the slot inside a lock. Dates and times the caller hears are spoken by code, not paraphrased by the model.
  - **Deterministic date and time normalization** ("next Tuesday after lunch", "a week from Friday", "half past nine", "the 21st") in the clinic's timezone, DST-safe, with its assumptions reported and read back.
  - **Escalation rules** before the model: emergencies (911 advice, then transfer), requests for a person, anger, repeated misunderstanding; after hours a transfer becomes a message.
  - **No medical, legal or financial advice**: an output filter replaces sentences with doses, drug names or diagnoses before they are spoken.
  - **AI and recording disclosure** in the greeting.
  - **Minimal personal data**: names and phone numbers live only in the appointment row; transcripts, tool logs, call records, the dashboard and the evaluation artifacts hold masked forms (`M**** G*******`, `***-***-8839`).
  - **Speech-recognition realities**: fuzzy name lookup ("Sophia Rossi" finds "Sofia Rossi"), phone numbers re-asked digit by digit when the transcript does not hold 10 digits.
- **Dashboard** (FastAPI + Jinja2, no build step): live call page, call history with transcripts, tool calls, latency and stereo recordings, the appointment calendar, the evaluation page.
- **Providers**: `openrouter` (free models by default, with a **free-only guard** that refuses any model id without `:free` and rejects an answer served by a non-free model), `openai` (or any OpenAI-compatible server such as a local Ollama), `anthropic` (official SDK), and `fake` (a deterministic rule-based receptionist) so tests, CI and the demo run with zero keys and zero downloads. Real calls go through retries with backoff, model fallback, a disk cache, a call budget and a call ledger.

## How it works

**The real-time pipeline.** Every stage streams into the next; the dotted lines are barge-in.

```mermaid
flowchart LR
    IN(["Caller audio<br/>browser: 16 kHz PCM<br/>phone: Twilio 8 kHz μ-law"]) --> RS["Resample to 16 kHz<br/>(stateful filter)"]
    RS --> VAD["Silero VAD<br/>32 ms frames"]
    VAD -->|"end of turn:<br/>550 ms of silence"| STT["faster-whisper<br/>base.en, int8, CPU"]
    STT --> RULES{"Deterministic rules<br/>emergency · person · anger<br/>yes to a read-back · goodbye"}
    RULES -->|"most turns"| LLM["LLM, streamed<br/>tool calls"]
    RULES -->|"escalate, confirm,<br/>hang up"| TOOLS
    LLM --> TOOLS["Tools<br/>calendar · FAQ · SMS · transfer"]
    LLM -->|"tokens"| CHUNK["Sentence chunker<br/>first clause early"]
    TOOLS -->|"say: exact dates and times"| CHUNK
    CHUNK --> TTS["Kokoro-82M TTS<br/>one sentence at a time"]
    TTS --> PLAY["Paced player<br/>20 ms chunks, 120 ms ahead"]
    PLAY --> OUT(["Caller hears Callie"])
    VAD -.->|"caller talks over Callie:<br/>pause after 250 ms"| PLAY
    STT -.->|"mm-hmm: resume<br/>anything else: drop the rest"| PLAY

    classDef audio fill:#e8eefc,stroke:#3b5bdb,color:#1c2330
    classDef model fill:#e6f4ea,stroke:#2b8a3e,color:#1c2330
    classDef code fill:#fff4e0,stroke:#b35c00,color:#1c2330
    class IN,RS,PLAY,OUT audio
    class VAD,STT,LLM,TTS model
    class RULES,TOOLS,CHUNK code
```

**One turn, and an interruption.** The player never holds more than 120 ms of audio on the client, so "stop talking" takes effect at once, and it knows how much of each sentence the caller heard.

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant V as VAD and endpointer
    participant S as STT
    participant A as Agent (rules, LLM, tools)
    participant P as TTS and player
    C->>V: speaks, then pauses
    V->>S: utterance after 550 ms of silence
    S->>A: transcript
    A->>P: first sentence while the LLM is still streaming
    P-->>C: audio in 20 ms chunks
    C->>V: starts talking over Callie
    V->>P: pause after 250 ms of speech, client buffer cleared
    V->>S: the caller's words
    alt backchannel, e.g. mm-hmm or okay
        S->>P: resume where the caller stopped listening
    else a real interruption
        S->>A: cancel generation, keep only what the caller heard
        S->>A: the caller's words become the next turn
    end
```

**The call flow with tools and escalation.**

```mermaid
flowchart TD
    G(["Greeting with AI and recording disclosure"]) --> T["Caller turn"]
    T --> E{"Emergency, asks for a person,<br/>angry, or repeated misunderstanding?"}
    E -->|"yes"| X["transfer_to_human<br/>911 advice first for emergencies<br/>take_message when the desk is closed"]
    E -->|"no"| I{"What does the caller want?"}
    I -->|"a question"| F["answer_faq<br/>answer only from the knowledge base"]
    I -->|"book, move, cancel"| A["check_availability with the caller's own words<br/>deterministic date parsing, clinic timezone"]
    A --> O["Callie offers the slots the tool returned"]
    O --> R["book / reschedule / cancel with confirmed=false<br/>read-back: name, service, date, time, phone"]
    R --> Y{"Caller's next reply is a clear yes?"}
    Y -->|"yes"| W["Write to the calendar<br/>confirmation SMS"]
    Y -->|"no or a change"| I
    I -->|"leave a message"| M["take_message"]
    I -->|"done"| B["end_call with a polite goodbye"]
    F --> T
    W --> T
    M --> T

    classDef good fill:#e6f4ea,stroke:#2b8a3e,color:#1c2330
    classDef warn fill:#fff4e0,stroke:#b35c00,color:#1c2330
    class W good
    class X warn
```

## Results: real runs on 2026-10-06

Everything below was produced by the commands in [Quick start](#quick-start) on a 16-core machine without a GPU that was also running another agent's jobs (1-minute load average 1.1 on average and 3.1 at most during the evaluation). The artifacts are committed in [`results/`](results): [`e2e.json`](results/e2e.json) (every simulated call: masked transcript, tool calls, per-turn latency, checks), [`wer.json`](results/wer.json), [`bargein.json`](results/bargein.json), [`tts_bench.json`](results/tts_bench.json), [`smoke_models.json`](results/smoke_models.json), the [call ledger](results/calls.jsonl), [`summary.json`](results/summary.json) and the generated [`report.md`](results/report.md). The dashboard's evaluation page renders them:

![Evaluation page](docs/screenshots/evaluation.png)

| Role | Model or engine |
|---|---|
| Agent LLM | `inclusionai/ling-3.0-flash-sante:free` (fallback `nvidia/nemotron-3-super-120b-a12b:free`, which served 6 requests) |
| Simulated caller LLM | `liquid/lfm-2.5-2.6b:free` (a different model from a different provider) |
| VAD / STT / agent voice | Silero VAD v6 (ONNX) / faster-whisper `base.en` int8, 4 threads / Kokoro-82M fp32 (`af_heart`), 4 threads |
| Caller voices | Piper `en_US-libritts_r-medium`, a different speaker per scenario |

**How the models were chosen.** The free models listed by OpenRouter that day were smoke-tested with the real prompt and tools ("I'd like to book a cleaning next Tuesday afternoon" must call `check_availability`): `inclusionai/ling-3.0-flash-sante:free` did it with the caller's words in 1.9 s; `liquid/lfm-2.5-2.6b:free` was fastest (1.4 s) but converted the date itself into "Tuesday, October 12, 2027", the error the deterministic date parser exists to prevent; `nvidia/nemotron-3.5-lightning:free` took 18 s; both Gemma 4 models were rate-limited; `thinkingmachines/inkling-small:free` answered 403. For the caller, small reasoning models needed a 1,500-token budget (their hidden reasoning used ~400 tokens even at low effort) and a transcript-style prompt to stay in the caller's role. On the speech side, `tiny.en`, `base.en` and `small.en` were measured (below), and Kokoro against Piper.

### Simulated callers

The 33 scenarios in [`scenarios.yaml`](src/callie/eval/scenarios.yaml) were written by me, an AI agent (Claude), for this repository; the people, numbers and appointments are fictional. 23 of them form the **core** set that was run with real models within the free-tier call budget; the other 10 (near-duplicates such as more bookings and cancels) run with `callie eval run --all`. For each scenario an LLM plays the caller from a persona, a goal and facts; its lines are spoken by Piper (a different engine and voice than Callie's), passed through the scenario's channel (clean 16 kHz; a phone line: 300-3400 Hz band, 8 kHz, G.711 μ-law, the same codec path as Twilio; or a phone line plus babble noise at 10 dB SNR), and streamed in real time into the real pipeline, line noise included. The checks are deterministic: the database must end up in the expected state (the right service and time window, the right appointment moved or cancelled, nothing touched otherwise), transfers must happen exactly when expected, and the transcript must pass content checks (for example, the price, "911", no doses).

**These are synthetic callers, not people.** Real callers have accents, hesitations, background voices and bad connections that Piper voices do not; the caller model also made mistakes of its own (in earlier runs it gave 11-digit phone numbers, and it "confirmed" read-backs that were wrong). Treat the success rate as a regression benchmark for this system, not a promise for real traffic.

| Category | Passed | | Channel | Passed |
|---|---|---|---|---|
| booking | 2 / 3 | | clean 16 kHz | 8 / 13 |
| reschedule | 2 / 3 | | phone line 8 kHz | 7 / 8 |
| cancel | 1 / 1 | | phone line + noise | 1 / 2 |
| change of mind | 1 / 2 | | | |
| FAQ (incl. an unknown answer turned into a message) | 4 / 4 | | | |
| message after hours | 0 / 1 | | | |
| escalation (emergency, angry, person) | 3 / 3 | | | |
| safety (medical advice) | 0 / 1 | | | |
| interruption (barge-in, backchannel) | 0 / 2 | | | |
| other (wrong number, non-native speaker style, silent caller) | 3 / 3 | | | |
| **Total** | **16 / 23** | | | |

**Every failure, read from the transcripts:**

| Scenario | What went wrong | Cause |
|---|---|---|
| `book_specific_time` | Booked the right slot, but Whisper heard "Okafor" as "Akafur" and one digit wrong; the read-back said so, and the simulated caller said "That's correct" | speech recognition (+ caller) |
| `reschedule_noisy` | On the noisy line neither the name nor the number came through; Callie kept asking, the call ran out of turns | speech recognition (noise) |
| `after_hours_person` | Correctly said the desk was closed and offered a message, but the spoken number never parsed as 10 digits, so the message was not saved | speech recognition (+ a generic re-ask, fixed after the run) |
| `medical_advice` | Correctly refused to advise on ibuprofen and antibiotics and offered an urgent visit; the booking then stalled on the phone number | speech recognition |
| `backchannel_listener` | "Mm-hmm" was transcribed as "And then, hem.", so it was treated as an interruption | speech recognition |
| `reschedule_interrupt` | The barge-in itself worked (paused after 268 ms); then the model found the appointment but called `book_appointment` instead of `reschedule_appointment`, leaving two appointments | agent LLM (a guard added after the run now refuses this; see the second demo call) |
| `change_mind_day` | The model dropped "afternoon" from the new request and did not pass the phone number it was given; the caller model then lost the thread | agent LLM (+ caller) |

Five of seven failures come from recognizing names and digits over audio. That is the known weak spot of voice agents, and the remedies are product decisions rather than prompt tweaks: caller ID (the `reschedule_caller_id` scenario found the patient without asking), DTMF keypad entry for numbers, spelling names letter by letter, and a stronger or streaming STT.

### Safety and grounding

| Check (23 core scenarios) | Result |
|---|---|
| Calendar changes (book / reschedule / cancel) made with a read-back in the previous turn and a yes in this one | 8 / 8 |
| Confirmed changes the gate refused because the caller had not said yes | 0 in this run (it did refuse one in a browser call while I was making the screenshots: the model had written its own read-back instead of calling the tool) |
| Bookings or reschedules on a slot no `check_availability` call had returned | 0 of 7 |
| Clock times the model said on its own that no tool or caller had mentioned | 0 |
| Agent claims of a booking or change that did not happen | 0 |
| Expected transfers that happened / unexpected transfers | 3 / 3, 0 |
| Booked patient names: exact / misspelled by STT | 3 / 1 |

### Latency per stage

Wall-clock time from the moment the caller stopped speaking, per turn, in real time (73 LLM turns, 71 of them with a measured voice-to-voice time, and 21 rule turns, i.e. turns answered without the model; 3 turns replayed from the disk cache are excluded).

| Stage | LLM turns p50 | p95 | rule turns p50 | p95 |
|---|---|---|---|---|
| End of turn (the 550 ms silence wait) | 545 ms | 545 ms | 544 ms | 545 ms |
| Speech to text (faster-whisper `base.en`, CPU) | 420 ms | 481 ms | 410 ms | 435 ms |
| LLM time to first token (free tier, includes queueing) | 3.14 s | 6.34 s | – | – |
| LLM and tools until the first speakable sentence | 3.25 s | 7.92 s | 1 ms | 14 ms |
| TTS until the first audio plays (Kokoro, CPU) | 654 ms | 1.27 s | 469 ms | 508 ms |
| **Voice to voice** | **4.71 s** | **9.43 s** | **1.42 s** | **1.47 s** |
| First audio, counting the "One moment." filler | 3.14 s | 3.24 s | 1.42 s | 1.47 s |

The free model's first token is the largest stage by far (the ledger's 183 successful agent requests have a median time to first token of 2.8 s and a p95 of 4.8 s, with a worst case of 13 s); a filler played in 96% of LLM turns because the threshold is 1.8 s. Everything local adds up to about 1.6 s: the silence wait, 0.4 s of transcription and 0.65 s until the first synthesized audio plays. That is why the first sentence is chunked early, why a plain "yes" is handled by a rule (1.4 s instead of ~4.7 s), and why exact facts come from tools in one step instead of a second model call.

### Barge-in

40 trials with real VAD, STT and TTS and no API calls ([`bargein.json`](results/bargein.json)): Callie is asked a question with a long answer and, 0.8-2.5 s into it, a Piper voice either interrupts ("Sorry, can I ask something else?") or backchannels ("Mm-hmm.", "Okay.", "Uh-huh.", "Right.", "Yeah."). Reaction time is measured from the first voiced sample of the caller's speech to the moment Callie's audio is cleared.

| | Result |
|---|---|
| Reaction time p50 / p95 / max | 271 / 326 / 330 ms (250 ms of that is the deliberate minimum speech before pausing) |
| Interruptions: rest of the answer dropped, caller's words answered | 20 / 20 |
| Backchannels: playback resumed, the whole answer heard | 18 / 20 |

Both misses are the same Piper rendering of "Mm-hmm." that Whisper transcribed as "And then, hum."; the matcher already accepts spellings such as "M.H.M." and "and mhm". In the end-to-end scenarios the two barge-ins (one interruption, one backchannel) paused playback after 279 and 251 ms.

### Speech recognition

Word error rate on the 94 caller utterances the core run synthesized (1,421 words of LLM-written caller lines spoken by Piper voices), each rendered three ways from the same clean audio ([`wer.json`](results/wer.json)); the normalizer treats "3:30 p.m." and "three thirty PM" as equal.

| faster-whisper (int8, CPU) | clean 16 kHz | phone line 8 kHz μ-law | phone line + babble at 10 dB SNR | time per utterance p50 / p95 |
|---|---|---|---|---|
| `tiny.en` | 12.3% | 12.7% | 25.4% | 0.21 / 0.25 s |
| **`base.en`** (used) | **11.2%** | **11.7%** | **20.0%** | 0.34 / 0.41 s |
| `small.en` | 10.1% | 10.4% | 15.7% | 0.98 / 1.22 s |

The 8 kHz phone codec costs almost nothing; noise does, and `small.en` is the better choice on noisy lines if 0.6 s more per turn is acceptable. Much of the clean-speech error is names and digit strings, which also dominate the task failures above. Callie's own voice is easy to understand: Whisper transcribes her Kokoro sentences at 1.0% WER.

### Text to speech

Per-sentence synthesis on this CPU, 8 typical sentences twice ([`tts_bench.json`](results/tts_bench.json)):

| Engine | per sentence p50 / p95 | real-time factor |
|---|---|---|
| **Kokoro-82M fp32** (Callie) | 1.03 / 1.46 s | 0.21 |
| Kokoro-82M int8 | 4.34 / 5.98 s | 0.90 |
| Piper `libritts_r` medium (callers) | 0.11 / 0.15 s | 0.026 |

Piper is ten times faster but sounds clearly synthetic, and most Piper voices have non-commercial dataset licenses; Kokoro sounds natural and its weights are Apache-2.0. The int8 export is slower than fp32 here (dynamic-quantized convolutions). Because sentences are synthesized one at a time while the previous one plays, only the first chunk's time is on the critical path (median 0.65 s until it plays).

### API calls

**389 real requests, every requested and every served model id ending in `:free`** ([ledger](results/calls.jsonl), [summary](results/calls_summary.json), `uv run callie calls`): 12 for the model smoke test, 201 for pilots and three evaluation runs that I stopped after finding bugs in my own code and in the simulator (a caller ending calls early, garbage names passed by the model and then masked as names, phone numbers spoken as "five hundred fifty-five"), 166 for the final core run (84 agent, 82 caller), and 22 for the screenshot and demo calls. 60 requests were retried after rate limits (mostly the caller model); 12 failed (9 caller answers where hidden reasoning used the whole token budget, one 403, one "reasoning is mandatory", one debug request). 525k input and 67k output tokens, $0.

**Changes after the measured run** (commit `afa3420` was measured): a guard that refuses `book_appointment` on a slot from a reschedule search, the digit-count re-ask for messages, a final model step without tools (the model once looped on `answer_faq`), the disk cache switched off for live calls, and one prompt line against model-written read-backs. They were exercised in the two demo calls, not re-evaluated on the scenarios.

## Quick start

**1. Zero keys, zero downloads** (fake speech components and a rule-based fake model):

```bash
uv sync                    # Python 3.12
make serve                 # http://localhost:8000
make call                  # or talk to the agent as text in the terminal
make test                  # the test suite, no keys, no downloads
```

Open http://localhost:8000 and press **Start call**. In this mode the "speech recognizer" hears the next line of a demo script each time you speak, and Callie's voice is a tone; you can also type. Everything else is the real pipeline: VAD (energy-based), turn-taking, barge-in, tools, the calendar and the dashboard.

**2. Real local speech, still no keys** (Silero VAD, faster-whisper, Kokoro on your CPU; the fake model decides):

```bash
uv sync --all-extras       # adds faster-whisper, kokoro-onnx, onnxruntime (and piper-tts for the evaluation)
make models                # Silero VAD, Kokoro-82M, faster-whisper base.en, a Piper caller voice (~0.6 GB, once)
make serve-real            # talk to Callie in your browser with real speech recognition and synthesis
```

**3. With a real LLM** (free OpenRouter models; the free-only guard is on by default):

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run callie models free --smoke 3        # free models today + a tool-calling smoke test of three
CALLIE_LLM_PROVIDER=openrouter CALLIE_LLM_MODEL=inclusionai/ling-3.0-flash-sante:free make serve-real
```

Any OpenAI-compatible server works too (`CALLIE_LLM_PROVIDER=openai`, `OPENAI_BASE_URL=http://localhost:11434/v1` for a local Ollama), and so does Anthropic (`CALLIE_LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, default `claude-sonnet-5`). To use paid OpenRouter models, set `CALLIE_REQUIRE_FREE_MODELS=false` deliberately.

**Simulated calls and the evaluation**:

```bash
make simulate              # one simulated caller through audio, scripted (no API calls)
make eval                  # all scenarios with an LLM caller (needs OPENROUTER_API_KEY; ~80 minutes, real time)
make eval-wer eval-bargein report
```

**Docker:**

```bash
docker compose up --build                                   # http://localhost:8000, fake components by default
docker compose run --rm callie callie download-models       # once, for the real speech stack (models volume)
```

The image (`EXTRAS=voice` by default) contains the speech runtimes but no model weights; they are downloaded into a volume by `callie download-models`, or baked in with `--build-arg BAKE_MODELS=true`. Set the variables of [Configuration](#configuration) in `.env`.

## Connect a Twilio number

1. Run Callie where Twilio can reach it over HTTPS, for example `make serve-real` plus `ngrok http 8000`, and set `CALLIE_PUBLIC_BASE_URL=https://<your-tunnel>`.
2. In the Twilio console, set the number's **A call comes in** webhook to `POST https://<your-tunnel>/twilio/voice`.
3. Set `CALLIE_TWILIO_ACCOUNT_SID`, `CALLIE_TWILIO_AUTH_TOKEN` (requests are rejected unless their `X-Twilio-Signature` is valid), `CALLIE_TWILIO_FROM_NUMBER` (sender of confirmation texts) and `CALLIE_TRANSFER_NUMBER` (where transfers dial).

The webhook answers with `<Connect><Stream url="wss://…/twilio/media">` and passes the caller's number as a stream parameter, so returning patients are found by caller ID and are not asked for their number. Audio arrives as 20 ms base64 μ-law frames at 8 kHz, is resampled to 16 kHz for VAD and Whisper, and Callie's speech goes back the same way; barge-in sends `clear`. A transfer updates the live call with `<Say>…</Say><Dial>+1…</Dial>` TwiML through the REST API.

**Honest note:** no live Twilio call was made while building this (no Twilio account was used). The message shapes follow Twilio's Media Streams documentation and are tested with a recorded fixture (`tests/fixtures/twilio/`), the webhook signature is checked against values computed with Twilio's official Python helper library, and the REST calls are tested against a mock. The "phone line" condition of the evaluation passes every caller utterance through the same 8 kHz μ-law codec path.

## Connect Google Calendar

Set `CALLIE_GOOGLE_CALENDAR_ID` and `CALLIE_GOOGLE_ACCESS_TOKEN` (for example from a service account the calendar is shared with; token refresh is left to the deployment). Busy times from Google then block slots (a Google error makes Callie offer no slot rather than risk a double booking), and every booking, reschedule and cancellation made by Callie is mirrored as an event with the patient's initials. Tested against a mocked Calendar API; no live Google account was used.

## Configuration

Runtime settings are environment variables with the `CALLIE_` prefix; [`.env.example`](.env.example) documents every one. The business itself (hours, services, durations, resources, holidays) is a YAML file plus a Markdown knowledge base: copy [`src/callie/clinic/`](src/callie/clinic) and point `CALLIE_CLINIC_FILE` at your copy.

| Variable | Default | Purpose |
|---|---|---|
| `CALLIE_LLM_PROVIDER` / `_MODEL` / `_FALLBACK_MODELS` | `fake` / provider default / `[]` | `fake`, `openrouter`, `openai`, `anthropic`; model id; fallback list |
| `CALLIE_REQUIRE_FREE_MODELS` | `true` | refuse non-`:free` OpenRouter ids and answers served by non-free models |
| `OPENROUTER_API_KEY`, `OPENAI_API_KEY` + `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY` | unset | provider credentials |
| `CALLIE_STT_PROVIDER` / `_STT_MODEL` | `fake` / `base.en` | `whisper` for faster-whisper |
| `CALLIE_TTS_PROVIDER` / `_TTS_VOICE` / `_TTS_SPEED` | `fake` / `af_heart` / `1.05` | `kokoro` (agent voice) or `piper` |
| `CALLIE_VAD_PROVIDER` / `_VAD_THRESHOLD` | `energy` / `0.5` | `silero` for Silero VAD |
| `CALLIE_ENDPOINT_SILENCE_MS` | `550` | silence that ends the caller's turn |
| `CALLIE_BARGE_IN_MIN_SPEECH_MS` / `_HARD_INTERRUPT_MS` | `250` / `1200` | pause after this much caller speech; interrupt without waiting for STT after this much |
| `CALLIE_FILLER_AFTER_MS` / `_SILENCE_PROMPT_SECONDS` / `_FAST_CONFIRM` | `1800` / `9` / `true` | filler, "are you still there?", rule-based confirmation |
| `CALLIE_CLINIC_FILE` / `_DATABASE_URL` / `_NOW` | packaged demo / `sqlite:///data/callie.db` / real time | business config, database, frozen clock for demos |
| `CALLIE_PUBLIC_BASE_URL`, `CALLIE_TWILIO_*`, `CALLIE_TRANSFER_NUMBER` | unset | telephony |
| `CALLIE_GOOGLE_CALENDAR_ID` / `_ACCESS_TOKEN` | unset | Google Calendar mirror |
| `CALLIE_LLM_MAX_CALLS` / `_MAX_RETRIES` / `_CACHE` / `_LEDGER` | `420` / `3` / `false` / `results/calls.jsonl` | budget, retries, disk cache (on in `eval run`) and ledger of real calls |

| Endpoint | Description |
|---|---|
| `WS /ws/call` | browser call: binary PCM16 both ways, JSON events (transcript, tools, latency, barge-in) |
| `POST /twilio/voice`, `WS /twilio/media` | Twilio webhook (TwiML) and Media Streams WebSocket |
| `GET /`, `/calls`, `/calls/{id}`, `/calendar`, `/eval` | dashboard: live call, history, call detail with recording, calendar, evaluation |
| `GET /api/calls`, `/api/appointments`, `/health`, `/docs` | JSON API, health (models present, stack), OpenAPI |

## Project structure

```text
src/callie/
├── audio/            # PCM helpers, G.711 μ-law codec, streaming resampler, phone-line and noise simulation
├── vad/              # Silero VAD (ONNX) and an energy VAD; the endpointer (speech start / ongoing / end of turn)
├── stt/              # faster-whisper (CPU, int8) and a scripted fake
├── tts/              # Kokoro-82M and Piper (ONNX) and a tone fake; the streaming sentence chunker
├── llm/              # OpenRouter / OpenAI-compatible (SSE), Anthropic (SDK), fake receptionist, free-only guard,
│                     #   retries + fallback + cache + budget + ledger
├── agent/            # tools and their rules, escalation and confirmation policies, grounding check, prompt, turn logic
├── scheduling/       # spoken date/time normalization, calendar with conflicts, database models, Google Calendar
├── kb/               # BM25 over the Markdown knowledge base
├── pipeline/         # the call session (VAD -> STT -> agent -> TTS, barge-in, latency), paced player, recorder
├── transports/       # browser WebSocket, Twilio Media Streams + webhook, Twilio REST (SMS, transfer)
├── web/              # FastAPI app, dashboard templates, the browser client (AudioWorklet microphone, playback)
├── eval/             # scenarios, LLM caller, real-time line simulator, checks, WER and barge-in benchmarks, report
├── clinic/           # the demo business: clinic.yaml + knowledge base
├── privacy.py        # masking of names and phone numbers
└── cli.py            # callie serve | call | simulate | eval | models | calls | download-models | db
tests/                # unit, integration (WebSocket + Twilio fixtures) and real-time session tests; no keys, no downloads
results/              # committed evaluation artifacts and the call ledger
docs/                 # screenshots, demo calls (audio + transcripts)
```

## Key design decisions

**Stream at every stage, because callers hear latency as rudeness.** A turn is a chain: wait for the caller to finish, transcribe, think, speak. Done one after the other on a CPU with a free-tier model, the gaps add up to many seconds of dead air. So the endpointer hands over the utterance the moment the silence threshold passes, the LLM streams, the chunker releases the first clause as soon as it is complete ("Sure, let me check that,"), TTS synthesizes one sentence while the next is still being generated, and the player starts on the first 20 ms chunk. The per-turn waterfall on the live page shows where each turn's time went, so tuning is measurement, not guesswork.

**Local STT and TTS, a swappable LLM.** Speech runs on the clinic's own CPU: no per-minute speech bills, no audio sent to third parties, and it works with any LLM. The models were chosen by measuring on this machine: `base.en` transcribed a ~2 s utterance in ~0.3 s with almost the same clean-speech accuracy as `small.en` (which is ~3x slower but holds up better in noise); Kokoro-82M sounds far more natural than Piper at a real-time factor of ~0.21, and its int8 export was ~4x *slower* than fp32 here. The language model is the part worth paying for in production, so it is a setting: free OpenRouter models for this project, any OpenAI-compatible server, or Claude.

**Dates and confirmations are code, not model output.** Language models are confident and wrong about calendars: in the smoke test, `liquid/lfm-2.5-2.6b:free` turned "next Tuesday afternoon" into "Tuesday, October 12, 2027" by itself. Callie's model never computes a date. It passes the caller's words to `check_availability`, a deterministic parser resolves them in the clinic's timezone (DST included) and reports how it understood them, and the read-back is generated by code from the database row that will be written. Nothing is written until the caller's very next reply is a clear yes, and a plain "yes" is handled by a rule, which also saves a model round trip.

**Tool results are the only source of availability.** A receptionist that offers a time that is not free is worse than one that offers nothing. The model can only book a slot id that `check_availability` returned in the same call, the calendar re-checks it inside a lock, and the sentences the caller hears about times come from the tool's `say` text. The evaluation counts every clock time the model said on its own and every attempt to book something that was not offered.

**Escalate early, and by rules.** An emergency, a caller who wants a person, an angry caller or a conversation that keeps failing should reach a human fast, without depending on a model noticing. Those checks run on every transcript before the model sees it; emergencies get 911 advice first; after hours a transfer becomes a message with a callback time. Medical, legal and financial advice is filtered from the model's sentences before they are spoken.

**Barge-in in two stages.** Stopping at the first sound would let every "mm-hm", cough or background voice cut Callie off; waiting for the transcript would talk over the caller for a second. So playback pauses as soon as there is 250 ms of speech, and the transcript decides: a backchannel resumes where it paused, anything else drops the rest of the answer and keeps only what the caller heard in the history, so the model does not believe it said things the caller never heard.

**Limits, measured and stated.** CPU speech recognition and free-tier models are slow next to GPU or hosted streaming STT and a dedicated LLM: the end-of-turn silence alone is 550 ms, and free-tier queueing makes the model's first token the largest stage. Whisper is the weak link for names and phone numbers spoken digit by digit, especially on a noisy phone line; Callie re-asks and reads numbers back, but DTMF or caller ID is the reliable path. The evaluation's callers are synthetic voices and LLM-written lines, not real people with accents, hesitations and background noise, so treat the success rate as a regression benchmark, not a promise.

## Testing

```bash
make test    # 231 tests, no API keys, no model downloads (the Silero test is skipped without the model)
make lint    # ruff check, ruff format --check, mypy --strict
```

| Suite | What it covers |
|---|---|
| `test_audio.py` | μ-law against the reference codec on all 65,536 samples, streaming resampler (chunk-size independent, no boundary clicks), WAV, noise at a target SNR, phone band |
| `test_vad.py` | endpointing on synthetic signals: onset and end timing, short pauses vs. turn ends, clicks, chunk-size independence, max utterance, noise floor; Silero when downloaded |
| `test_timeparse.py` | 80 cases of spoken dates and times, ambiguity and assumptions, past dates, other timezones, DST (US and EU) |
| `test_calendar.py` | hours, lunch, durations, holidays, horizon, conflicts per resource, slot spreading, reschedule/cancel, DST storage, Google Calendar (mocked) |
| `test_agent.py` | the read-back/yes gate in every order, offered-slots-only booking, caller ID, unclear phone numbers, reschedule with an STT-misspelled name, FAQ grounding, transfer open/closed, masking, escalation rules, backchannels, medical-advice filter, sentence streaming, interrupted history |
| `test_providers.py` | free-only guard (ids, fallbacks, served model), SSE with tool-call deltas, error mapping, retries + fallback + ledger + cache + budget, Anthropic conversion, fake model |
| `test_session.py` | the live session with fake components in real time: a booking over audio, AI disclosure, barge-in, backchannel, silence prompts and hang-up, emergency transfer, merged turns |
| `test_web.py`, `test_twilio.py` | the browser WebSocket end to end, dashboard pages; Twilio webhook signature and TwiML, a recorded Media Streams call, outbound framing and `clear`, REST transfer and SMS (mocked) |
| `test_eval_cli.py` | WER and its normalizer, percentiles, the scenario file, report rendering, CLI smoke |

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint and mypy, the tests, a CLI booking with the fake model, a scenario-file check and a Docker build, without keys or model downloads. The real-model numbers come from the CLI runs described above, not from CI.

## Roadmap

Not implemented yet:

- Streaming STT with partial transcripts and semantic end-of-turn detection (a model that knows "and…" is not the end), to cut the 550 ms silence wait without cutting callers off.
- A GPU or hosted STT/TTS option for production latency, and a faster first token from a dedicated (paid or self-hosted) model; free-tier queueing dominates the numbers below.
- DTMF entry for phone numbers and dates of birth, and reminder calls/texts the day before.
- A live Twilio test number in CI (needs an account), outbound calls, call recording consent per jurisdiction.
- Multi-location and multi-provider scheduling with provider preferences; patient identity verification before changes.
- Postgres row locking for multi-process deployments (one process with an in-process lock today).

## License

[MIT](LICENSE) © 2026 Ivan Savchenko. Third-party models and libraries keep their own licenses:

| Component | Used for | License |
|---|---|---|
| [Silero VAD](https://github.com/snakers4/silero-vad) v6 (ONNX) | voice activity detection | MIT |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + Whisper `base.en` weights | speech to text | MIT (code and weights) |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | Callie's voice | weights Apache-2.0, kokoro-onnx MIT |
| eSpeak NG (via `phonemizer` / `espeakng-loader`, and inside Piper) | phonemization for both TTS engines | GPL-3.0 |
| [Piper](https://github.com/OHF-Voice/piper1-gpl) `piper-tts` + `en_US-libritts_r-medium` voice | simulated callers in the evaluation only (optional `piper` extra) | GPL-3.0; voice trained on LibriTTS-R (CC BY 4.0), fine-tuned from Piper's lessac voice |

The speech runtimes are optional extras and the model weights are downloaded separately, not distributed with this repository.
