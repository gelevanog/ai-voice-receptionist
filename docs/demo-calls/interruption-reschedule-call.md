# Demo call 2: rescheduling, with the caller interrupting Callie

Audio: [interruption-reschedule-call.mp3](interruption-reschedule-call.mp3) (stereo: caller left, Callie right). A browser call placed in headless Chrome: the caller is a scripted line-by-line synthetic voice (Piper) played as the microphone; Callie runs the real stack (Silero VAD, faster-whisper base.en, `inclusionai/ling-3.0-flash-sante:free`, Kokoro-82M) on October 6, 2026. Transcribed from the recording with faster-whisper small.en, so a few words may be misheard ("Sophie" is the fictional Sofia Rossi). The caller cuts Callie off at 00:39 while she lists Thursday's times; Callie stops within ~250 ms and answers the new request. The appointment Callie found (8 AM) is a booking for the same fictional name that an evaluation run had left in the demo database.

- `00:01.7` **Callie:** Thanks for calling Bright Side Dental. I'm Callie, the clinic's AI assistant, and this call may be recorded. How can I help you today?
- `00:11.9` **Caller:** Hi, it's Sophie, or Rossi. I have a cleaning tomorrow afternoon and I need to move it.
- `00:18.4` **Callie:** ONE MOMENT
- `00:20.4` **Callie:** Let me look up your appointment. Sophie, one moment please. I found your cleaning and checkup on Wednesday, October 7th at 8am. What day and time would you like instead?
- `00:30.2` **Caller:** Do you have anything on Thursday?
- `00:34.5` **Callie:** Let me check. I have Thursday, October 8, at 9.30 a.m., 10 a.m. or 10 a.m.
- `00:39.0` **Caller:** Sorry, sorry, in the morning, please.
- `00:43.5` **Callie:** Sure, one second. No problem. I have Thursday morning at 9 30 a.m. 10 a.m. Or 10 30 a.m. Which one works for you?
- `00:56.3` **Caller:** The first one works.
- `01:00.4` **Callie:** One moment.
- `01:03.4` **Callie:** Just to confirm, I'll move your cleaning and check up from Wednesday, October 7th at 8 a.m. to Thursday, October 8th at 9, 30 a.m. Is that right?
- `01:17.2` **Caller:** Yes, that's right.
- `01:19.0` **Callie:** Done. Your cleaning and check-up is now on Thursday, October 8th at 9, 30 a.m. I've texted you the new time. Anything else?
- `01:32.8` **Caller:** No, that's all thanks. Bye
- `01:35.5` **Callie:** Thanks for calling Bright Side Dental. Have a great day. Goodbye.
