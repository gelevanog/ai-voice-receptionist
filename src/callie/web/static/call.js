// Live browser call: microphone up, agent audio down, transcript, tool calls and a per-turn latency waterfall.
(() => {
  const $ = (s) => document.querySelector(s);
  const transcript = $("#transcript");
  const waterfall = $("#waterfall");
  const tip = $("#tip");
  let ws = null, ctx = null, stream = null, node = null;
  let outRate = 24000, nextTime = 0, sources = [];
  const agentBubbles = {};
  const turns = [];

  function esc(text) { return String(text ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]); }
  function scroll() { transcript.scrollTop = transcript.scrollHeight; }
  function add(html) { transcript.insertAdjacentHTML("beforeend", html); scroll(); return transcript.lastElementChild; }
  function setState(state, label) {
    const el = $("#state");
    el.dataset.state = state;
    el.querySelector("span:last-child").textContent = label || state.replace("_", " ");
  }

  function agentText(event) {
    let bubble = agentBubbles[event.turn];
    if (!bubble || bubble.dataset.closed) {
      bubble = add(`<div class="msg agent"><span class="who">Callie</span><span class="text"></span></div>`);
      agentBubbles[event.turn] = bubble;
    }
    const span = bubble.querySelector(".text");
    const piece = event.source === "filler" ? `<span class="muted">${esc(event.text)}</span>` : esc(event.text);
    span.innerHTML += (span.innerHTML ? " " : "") + piece;
    scroll();
  }

  function compact(value) {
    if (!value || typeof value !== "object") return esc(value);
    const parts = Object.entries(value).filter(([, v]) => v !== "" && v !== null && v !== undefined && !(Array.isArray(v) && !v.length))
      .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`);
    return esc(parts.join("  "));
  }

  function tool(event) {
    if (agentBubbles[event.turn]) agentBubbles[event.turn].dataset.closed = "1";
    const result = event.result || {};
    const slots = (result.slots || result.alternatives || []).map((s) => s.time).join("; ");
    const summary = [result.status, result.understood_as && `understood: ${result.understood_as}`, slots && `slots: ${slots}`,
      result.appointment_id && `id ${result.appointment_id}`].filter(Boolean).join(" · ");
    add(`<div class="tool"><span class="name">${esc(event.name)}</span> <span class="muted">${event.ms ?? ""} ms</span>
      <div class="args">${compact(event.arguments)}</div><div class="res">→ ${esc(summary)}</div></div>`);
  }

  const STAGES = [["endpointing_ms", "st-end", "End of turn"], ["stt_ms", "st-stt", "STT"], ["agent_ms", "st-llm", "LLM + tools"], ["tts_ms", "st-tts", "TTS + send"]];
  function renderWaterfall() {
    const max = Math.max(2000, ...turns.map((t) => t.voice_to_voice_ms || 0));
    const step = max > 8000 ? 2000 : max > 4000 ? 1000 : 500;
    const ticks = [];
    for (let v = 0; v <= max; v += step) ticks.push(`<span>${(v / 1000).toFixed(v % 1000 ? 1 : 0)}s</span>`);
    waterfall.innerHTML = turns.map((t) => {
      const segs = STAGES.map(([key, cls, label]) => {
        const v = Math.max(0, t[key] || 0);
        return `<span class="${cls}" style="width:${(v / max) * 100}%" data-tip="${label}: ${v} ms"></span>`;
      }).join("");
      const meta = [t.path === "fast_confirm" ? "confirmed by rules (no LLM call)" : t.path === "rules" ? "rules" : t.model,
        t.llm_first_token_ms != null ? `first token ${t.llm_first_token_ms} ms` : null, t.cached ? "cached" : null,
        t.filler ? `filler at ${t.first_audio_ms} ms` : null, t.interrupted ? "interrupted" : null].filter(Boolean).join(" · ");
      return `<div class="wf-row"><span class="muted">#${t.turn}</span><div class="wf-bar">${segs}</div>
        <span class="wf-total">${t.voice_to_voice_ms != null ? (t.voice_to_voice_ms / 1000).toFixed(2) + " s" : "–"}</span>
        <div class="wf-meta">${esc(meta)}</div></div>`;
    }).join("") + (turns.length ? `<div class="wf-axis"><div></div><div>${ticks.join("")}</div><div></div></div>` : "");
    const v2v = turns.map((t) => t.voice_to_voice_ms).filter((v) => v != null).sort((a, b) => a - b);
    $("#v2v-median").textContent = v2v.length ? (v2v[Math.floor((v2v.length - 1) / 2)] / 1000).toFixed(2) + " s" : "–";
  }
  waterfall.addEventListener("mousemove", (e) => {
    const target = e.target.closest("[data-tip]");
    if (!target) { tip.style.display = "none"; return; }
    tip.textContent = target.dataset.tip;
    tip.style.display = "block";
    tip.style.left = e.clientX + 12 + "px";
    tip.style.top = e.clientY + 12 + "px";
  });
  waterfall.addEventListener("mouseleave", () => { tip.style.display = "none"; });

  function onEvent(event) {
    switch (event.type) {
      case "call_started":
        $("#call-id").textContent = event.call_id;
        break;
      case "audio_format": outRate = event.rate; break;
      case "clear":
        sources.forEach((s) => { try { s.stop(); } catch (_) { /* already stopped */ } });
        sources = [];
        nextTime = ctx ? ctx.currentTime : 0;
        break;
      case "state":
        setState(event.state, { caller_speaking: "caller speaking", transcribing: "transcribing", thinking: "thinking", speaking: "Callie speaking", listening: "listening" }[event.state]);
        break;
      case "transcript": {
        if (event.merged) {
          const last = [...transcript.querySelectorAll(".msg.caller .text")].pop();
          if (last) { last.textContent = last.textContent + " " + event.text; break; }
        }
        add(`<div class="msg caller"><span class="who">Caller</span><span class="text">${esc(event.text)}</span></div>`);
        break;
      }
      case "agent_text": agentText(event); break;
      case "tool": tool(event); break;
      case "metrics":
        turns.push({ ...event, agent_ms: event.agent_ms, tts_ms: event.tts_ms });
        renderWaterfall();
        break;
      case "barge_in":
        if (event.stage === "paused") add(`<div class="event barge">caller started talking: Callie paused after ${event.reaction_ms} ms</div>`);
        else if (event.stage === "resumed") add(`<div class="event">backchannel “${esc(event.text)}”: Callie continues</div>`);
        else add(`<div class="event barge">barge-in: rest of the answer dropped</div>`);
        if (event.stage === "interrupted") Object.values(agentBubbles).slice(-1).forEach((b) => b.classList.add("interrupted"));
        break;
      case "action":
        add(`<div class="event">${event.action === "transfer" ? "transferring to the front desk" : "Callie ended the call"}${event.reason ? ": " + esc(event.reason) : ""}</div>`);
        break;
      case "transfer": add(`<div class="event">${esc(event.note)}</div>`); break;
      case "call_ended":
        add(`<div class="event">call ended · outcome <b>${esc(event.outcome)}</b> · <a href="/calls/${esc(event.call_id)}">call record</a></div>`);
        stop(false);
        break;
    }
  }

  function playPcm(buffer) {
    if (!ctx) return;
    const pcm = new Int16Array(buffer);
    const audioBuffer = ctx.createBuffer(1, pcm.length, outRate);
    const data = audioBuffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) data[i] = pcm[i] / 32768;
    const src = ctx.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(ctx.destination);
    const at = Math.max(ctx.currentTime + 0.03, nextTime);
    src.start(at);
    nextTime = at + audioBuffer.duration;
    sources.push(src);
    src.onended = () => { sources = sources.filter((s) => s !== src); };
  }

  async function start() {
    $("#start").disabled = true;
    transcript.innerHTML = "";
    turns.length = 0; renderWaterfall();
    Object.keys(agentBubbles).forEach((k) => delete agentBubbles[k]);
    ctx = new AudioContext({ latencyHint: "interactive" });
    await ctx.audioWorklet.addModule("/static/mic-worklet.js");
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    } catch (err) {
      add(`<div class="event">Microphone not available (${esc(err.message)}). You can type below.</div>`);
    }
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/call`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (m) => (typeof m.data === "string" ? onEvent(JSON.parse(m.data)) : playPcm(m.data));
    ws.onclose = () => stop(false);
    ws.onopen = () => {
      $("#hangup").disabled = false;
      $("#say").disabled = false;
      setState("listening", "connected");
      if (stream) {
        const source = ctx.createMediaStreamSource(stream);
        node = new AudioWorkletNode(ctx, "mic-16k");
        node.port.onmessage = (e) => {
          if (ws && ws.readyState === 1) ws.send(e.data.pcm);
          $("#meter").style.width = Math.min(100, e.data.level * 400) + "%";
        };
        source.connect(node);
      }
    };
  }

  function stop(sendHangup = true) {
    if (ws && ws.readyState === 1 && sendHangup) ws.send(JSON.stringify({ type: "hangup" }));
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (node) node.disconnect();
    stream = null; node = null;
    $("#start").disabled = false;
    $("#hangup").disabled = true;
    $("#say").disabled = true;
    setState("idle", "not connected");
  }

  $("#start").addEventListener("click", start);
  $("#hangup").addEventListener("click", () => stop(true));
  $("#typed").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = $("#typed input");
    if (ws && ws.readyState === 1 && input.value.trim()) {
      ws.send(JSON.stringify({ type: "text", text: input.value.trim() }));
      input.value = "";
    }
  });
  window.callie = { start, stop };
})();
