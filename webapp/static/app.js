/* Feelspace — front-end logic. No framework, no build step. */
(() => {
  "use strict";

  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ------------------------------------------------------------------
  // Content
  // ------------------------------------------------------------------
  const EMO = {
    neutral:   { label: "Neutral",    tone: "Steady & even",      score: 60, headline: "Your voice sounds steady.",
                 message: "Nothing stands out strongly — an even, level tone. A good moment to ask yourself what you need next." },
    calm:      { label: "Calm",       tone: "Relaxed & settled",  score: 80, headline: "You sound calm and settled.",
                 message: "Your voice is soft and unhurried. Whatever you're doing seems to be working — try to carry it into the rest of your day." },
    happy:     { label: "Happy",      tone: "Bright & upbeat",    score: 88, headline: "You sound bright today.",
                 message: "There's warmth and lift in your voice. Notice what's going well — naming good moments helps them last." },
    sad:       { label: "Low",        tone: "Low & heavy",        score: 22, headline: "Your voice sounds a little low.",
                 message: "Today might feel heavy, and that's okay. Small, kind steps count — and you don't have to carry it alone." },
    angry:     { label: "Frustrated", tone: "Tense & forceful",   score: 30, headline: "You sound tense or frustrated.",
                 message: "There's force and tension in your voice. Frustration is a signal, not a flaw — let your body release some of it first." },
    fearful:   { label: "Anxious",    tone: "Worried & on edge",  score: 26, headline: "You sound a bit anxious.",
                 message: "Your voice carries some worry. Slowing your breathing is the quickest way to tell your nervous system you're safe." },
    disgust:   { label: "Irritated",  tone: "Put off & uneasy",   score: 36, headline: "You sound irritated or put off.",
                 message: "Something seems to be bothering you. Naming it — even just to yourself — can take away some of its weight." },
    surprised: { label: "Energised",  tone: "Lively & alert",     score: 70, headline: "You sound energised.",
                 message: "There's a lot of lift in your voice — excitement or surprise. If it feels buzzy, a short pause can help you focus it." },
  };
  const LOW = new Set(["sad", "fearful", "angry", "disgust"]);

  const ICON = {
    wind: '<svg viewBox="0 0 24 24"><path d="M3 8h11a3 3 0 1 0-3-3M3 12h16a3 3 0 1 1-3 3M3 16h7"/></svg>',
    hand: '<svg viewBox="0 0 24 24"><path d="M7 11V5.5a1.5 1.5 0 0 1 3 0V11M10 10V4a1.5 1.5 0 0 1 3 0v6M13 10V5a1.5 1.5 0 0 1 3 0v7M16 9.5a1.5 1.5 0 0 1 3 0V14a7 7 0 0 1-7 7h-1a6 6 0 0 1-5-2.7L3.5 15a1.6 1.6 0 0 1 2.6-1.8L7 14.5"/></svg>',
    pen: '<svg viewBox="0 0 24 24"><path d="M4 20h4L19 9a2.8 2.8 0 0 0-4-4L4 16zM13.5 6.5l4 4"/></svg>',
    chat: '<svg viewBox="0 0 24 24"><path d="M4 5h16v11H9l-5 4z"/></svg>',
    walk: '<svg viewBox="0 0 24 24"><circle cx="13" cy="4.5" r="1.8"/><path d="M10 21l2-6 3 3v3M8.5 12.5l1.5-4 4 1 2 3.5M12 15l-1.5-6.5"/></svg>',
    sun: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
    book: '<svg viewBox="0 0 24 24"><path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2zM4 19V5M8 7h7"/></svg>',
    music: '<svg viewBox="0 0 24 24"><path d="M9 18V5l11-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="17" cy="16" r="3"/></svg>',
    water: '<svg viewBox="0 0 24 24"><path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11Z"/></svg>',
    heart: '<svg viewBox="0 0 24 24"><path d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10Z"/></svg>',
    target: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r=".6"/></svg>',
    moon: '<svg viewBox="0 0 24 24"><path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5Z"/></svg>',
  };

  // action: { label, go: tab, breathe?: technique, journal?: prompt }
  const RECS = {
    sad: [
      { icon: "chat", title: "Reach out to one person", text: "Send a short message to a friend or family member — even “hey, can we talk later?” is enough. Connection is the strongest buffer against low mood.", action: { label: "See support options", go: "support" } },
      { icon: "walk", title: "Ten minutes outside", text: "Daylight and gentle movement lift mood within minutes. Walk somewhere without a destination, phone in your pocket.", },
      { icon: "pen", title: "Name what's heavy", text: "Write one sentence about what's weighing on you, and one small thing that might make tonight a little easier.", action: { label: "Write in journal", journal: true } },
    ],
    fearful: [
      { icon: "wind", title: "4-7-8 breathing", text: "A longer out-breath switches on your body's calming system. Two minutes is enough to notice a difference.", action: { label: "Start breathing", go: "breathe", breathe: "478" } },
      { icon: "hand", title: "Ground yourself", text: "Racing thoughts? Use 5-4-3-2-1: notice what you can see, touch, hear, smell and taste.", action: { label: "Try grounding", go: "support", ground: true } },
      { icon: "target", title: "Shrink the worry", text: "Write the worry down, then the very next small step you can control. Exams and deadlines feel smaller in pieces.", action: { label: "Write in journal", journal: true } },
    ],
    angry: [
      { icon: "wind", title: "Box breathing", text: "Breathe in for 4, hold 4, out 4, hold 4. It slows your heart rate before you respond to anyone.", action: { label: "Start breathing", go: "breathe", breathe: "box" } },
      { icon: "walk", title: "Move it out", text: "A brisk walk, stairs, or a few stretches burns off the adrenaline that frustration brings." },
      { icon: "pen", title: "Unsent message", text: "Write exactly what you'd like to say — then don't send it. Getting it out of your head is the point.", action: { label: "Write in journal", journal: true } },
    ],
    disgust: [
      { icon: "pen", title: "Name the bother", text: "What exactly is bothering you? Putting a name to it (“I feel ___ because ___”) reduces its intensity.", action: { label: "Write in journal", journal: true } },
      { icon: "wind", title: "Reset with your breath", text: "A few slow, even breaths create a pause between the feeling and what you do next.", action: { label: "Start breathing", go: "breathe", breathe: "coherent" } },
      { icon: "music", title: "Change the scene", text: "Step away for five minutes — another room, some music you like, a glass of water." },
    ],
    neutral: [
      { icon: "target", title: "Plan one focused block", text: "Pick one task and give it 25 minutes of full attention, then a 5-minute break away from screens." },
      { icon: "water", title: "Quick body check", text: "When did you last drink water, eat, or stretch? Small physical needs quietly shape how we feel." },
      { icon: "pen", title: "One line for today", text: "Write a single sentence about today. Over a few weeks, these lines show you your own patterns.", action: { label: "Write in journal", journal: true } },
    ],
    calm: [
      { icon: "book", title: "Use the calm well", text: "Calm is a great state for deep work — tackle the task you've been putting off while it feels manageable." },
      { icon: "pen", title: "Notice what helped", text: "What made today feel calm? Saving it helps you return to it on harder days.", action: { label: "Write in journal", journal: true } },
      { icon: "moon", title: "Protect your sleep", text: "Keep the calm going into the evening — dim screens 30 minutes before bed." },
    ],
    happy: [
      { icon: "sun", title: "Savour it", text: "Spend 20 seconds really noticing this good feeling. Savouring makes positive moments stick.", },
      { icon: "pen", title: "Three good things", text: "Write down three things that went well today and why. It's one of the best-studied wellbeing habits.", action: { label: "Write in journal", journal: true } },
      { icon: "heart", title: "Share it", text: "Tell someone about something good — or check in on a friend who might be having a harder week." },
    ],
    surprised: [
      { icon: "wind", title: "Channel the energy", text: "If it feels buzzy, five slow breaths help turn restless energy into focus.", action: { label: "Start breathing", go: "breathe", breathe: "coherent" } },
      { icon: "target", title: "Ride the momentum", text: "High energy is great for starting things. Pick the one task you've been avoiding and begin it now." },
      { icon: "pen", title: "Capture the moment", text: "What just happened? Note it down while it's fresh.", action: { label: "Write in journal", journal: true } },
    ],
  };

  const PROMPTS = [
    "How has your day been so far — what stood out?",
    "What's taking up most of your mind right now?",
    "Describe how your body feels at this moment.",
    "What's one thing you're looking forward to, and one thing you're dreading?",
    "How did you sleep last night, and how is your energy?",
    "Talk about your week of classes or exams so far.",
    "Who did you talk to today, and how did it feel?",
    "If today had a weather forecast, what would it be?",
  ];

  const AFFIRMATIONS = [
    "You are more than your grades. Rest is part of the work, not a break from it.",
    "Progress is still progress, even on the days it's small.",
    "It's okay to ask for help. Strong people do it all the time.",
    "You don't have to feel ready to begin. Starting is what makes you ready.",
    "One difficult day is not a difficult life.",
    "Be as kind to yourself as you would be to a friend in your place.",
    "Breathe. You've handled hard things before, and you're still here.",
  ];

  const SELF_MOODS = [
    { id: "happy", label: "Good" },
    { id: "calm", label: "Calm" },
    { id: "neutral", label: "Okay" },
    { id: "sad", label: "Low" },
    { id: "fearful", label: "Anxious" },
    { id: "angry", label: "Frustrated" },
  ];

  const TECHNIQUES = {
    box: { name: "Box breathing", desc: "In 4 · hold 4 · out 4 · hold 4 — steadies focus before exams.", phases: [["Breathe in", 4, "in"], ["Hold", 4, "hold"], ["Breathe out", 4, "out"], ["Hold", 4, "hold"]] },
    "478": { name: "4-7-8 breathing", desc: "In 4 · hold 7 · out 8 — eases anxiety and helps with sleep.", phases: [["Breathe in", 4, "in"], ["Hold", 7, "hold"], ["Breathe out", 8, "out"]] },
    coherent: { name: "Coherent breathing", desc: "In 5 · out 5 — a gentle, even rhythm for everyday calm.", phases: [["Breathe in", 5, "in"], ["Breathe out", 5, "out"]] },
  };

  const GROUND = [
    [5, "Look around and name <strong>five things you can see</strong>."],
    [4, "Notice <strong>four things you can feel</strong> — your feet, the chair, your clothes, the air."],
    [3, "Listen for <strong>three things you can hear</strong>, near and far."],
    [2, "Find <strong>two things you can smell</strong> — or two scents you like."],
    [1, "Name <strong>one thing you can taste</strong>, or one thing you like about yourself."],
  ];

  // ------------------------------------------------------------------
  // Storage (wrapped: private windows may block it)
  // ------------------------------------------------------------------
  const store = {
    get(key, fallback) { try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); } catch { return fallback; } },
    set(key, val) { try { localStorage.setItem(key, JSON.stringify(val)); return true; } catch { return false; } },
  };
  const JOURNAL_KEY = "feelspace.journal.v1";
  const journal = () => store.get(JOURNAL_KEY, []);

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  const emoColor = (e) => `var(--e-${EMO[e] ? e : "neutral"})`;
  const pct = (x) => `${Math.round(x * 100)}%`;
  let toastTimer;
  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), 2600);
  }
  const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const dayKey = (d) => `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;

  // ------------------------------------------------------------------
  // Theme
  // ------------------------------------------------------------------
  // Light blue theme by default (even if the computer is in dark mode); the toggle switches to dark.
  document.documentElement.dataset.theme = store.get("feelspace.theme", "light");
  $("#themeToggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    store.set("feelspace.theme", next);
    drawWaveIdle();
  });

  // ------------------------------------------------------------------
  // Router
  // ------------------------------------------------------------------
  const TABS = ["home", "checkin", "breathe", "journal", "support", "about"];
  function route() {
    const tab = TABS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "home";
    TABS.forEach((t) => { $(`#view-${t}`).hidden = t !== tab; });
    $$("[data-tab]").forEach((a) => a.classList.toggle("active", a.dataset.tab === tab));
    $$("[data-tab]").forEach((a) => a.toggleAttribute("aria-current", a.dataset.tab === tab));
    if (tab === "journal") renderJournal();
    if (tab !== "breathe") stopBreathing();
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", route);

  // ------------------------------------------------------------------
  // Check-in: greeting, self mood, prompts
  // ------------------------------------------------------------------
  const h = new Date().getHours();
  $("#greeting").textContent = h < 5 ? "Still up? Let's check in." : h < 12 ? "Good morning." : h < 17 ? "Good afternoon." : "Good evening.";
  $("#dateLine").textContent = new Date().toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" });

  let selfMood = null;
  $("#selfMood").innerHTML = SELF_MOODS.map((m) =>
    `<button class="chip" role="radio" aria-checked="false" data-mood="${m.id}"><span class="dot" style="background:${emoColor(m.id)}"></span>${m.label}</button>`).join("");
  $("#selfMood").addEventListener("click", (e) => {
    const b = e.target.closest("[data-mood]");
    if (!b) return;
    selfMood = selfMood === b.dataset.mood ? null : b.dataset.mood;
    $$("#selfMood .chip").forEach((c) => c.setAttribute("aria-checked", String(c.dataset.mood === selfMood)));
  });

  let promptIdx = Math.floor(Math.random() * PROMPTS.length);
  const showPrompt = () => { $("#promptText").textContent = PROMPTS[promptIdx]; };
  $("#shufflePrompt").addEventListener("click", () => { promptIdx = (promptIdx + 1) % PROMPTS.length; showPrompt(); });
  showPrompt();
  $("#affirmation").textContent = AFFIRMATIONS[new Date().getDate() % AFFIRMATIONS.length];
  $("#homeQuote").textContent = `“${AFFIRMATIONS[(new Date().getDate() + 3) % AFFIRMATIONS.length]}”`;

  // ------------------------------------------------------------------
  // Models
  // ------------------------------------------------------------------
  let models = [];
  let selectedModel = store.get("feelspace.model", null);
  const prettyModel = (n) => n.replace(/_/g, " ").replace("wav2vec2", "wav2vec2").replace("6class", "· 6 emotions");
  async function loadModels() {
    try {
      const r = await fetch("/api/models");
      const j = await r.json();
      models = j.models || [];
      if (!models.find((m) => m.name === selectedModel)) selectedModel = j.default;
    } catch {
      models = [];
      selectedModel = null;
    }
    const sel = $("#modelSelect");
    sel.innerHTML = models.length
      ? models.map((m) => `<option value="${esc(m.name)}"${m.name === selectedModel ? " selected" : ""}>${esc(prettyModel(m.name))}</option>`).join("")
      : `<option>No model available</option>`;
    sel.disabled = !models.length;
    $("#modelList").innerHTML = models.length
      ? models.map((m) => `<li><span>${esc(prettyModel(m.name))} <span class="muted small">· ${m.labels.length} emotions · ${m.kind}</span></span><span>${m.accuracy != null ? pct(m.accuracy) + " test" : "—"}</span></li>`).join("")
      : `<li>The analysis server isn't running, or no trained model was found in <code>saved_models/</code>.</li>`;
    $("#modelPill").textContent = selectedModel ? `model: ${prettyModel(selectedModel)}` : "server offline";
  }
  $("#modelSelect").addEventListener("change", (e) => {
    selectedModel = e.target.value;
    store.set("feelspace.model", selectedModel);
    $("#modelPill").textContent = `model: ${prettyModel(selectedModel)}`;
    toast(`Using ${prettyModel(selectedModel)}`);
  });

  // ------------------------------------------------------------------
  // Recording (raw PCM -> WAV in the browser, so the server needs no ffmpeg)
  // ------------------------------------------------------------------
  const MAX_S = 15, MIN_S = 3;
  const rec = { stream: null, ctx: null, proc: null, analyser: null, chunks: [], sr: 44100, start: 0, raf: 0, timer: 0 };
  let lastBlobUrl = null;

  const canvas = $("#wave");
  const cctx = canvas.getContext("2d");
  function sizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
  }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function drawWaveIdle() {
    sizeCanvas();
    const { width: w, height: hgt } = canvas;
    cctx.clearRect(0, 0, w, hgt);
    cctx.strokeStyle = cssVar("--line");
    cctx.lineWidth = 2 * (window.devicePixelRatio || 1);
    cctx.beginPath();
    for (let x = 0; x <= w; x += 4) {
      const y = hgt / 2 + Math.sin(x / 22) * hgt * 0.06;
      x === 0 ? cctx.moveTo(x, y) : cctx.lineTo(x, y);
    }
    cctx.stroke();
  }
  function drawWaveLive() {
    const { width: w, height: hgt } = canvas;
    const buf = new Uint8Array(rec.analyser.fftSize);
    rec.analyser.getByteTimeDomainData(buf);
    cctx.clearRect(0, 0, w, hgt);
    cctx.strokeStyle = cssVar("--warm");
    cctx.lineWidth = 2.2 * (window.devicePixelRatio || 1);
    cctx.lineJoin = "round";
    cctx.beginPath();
    const step = buf.length / w;
    for (let x = 0; x < w; x += 2) {
      const v = (buf[Math.floor(x * step)] - 128) / 128;
      const y = hgt / 2 + v * hgt * 0.9;
      x === 0 ? cctx.moveTo(x, y) : cctx.lineTo(x, y);
    }
    cctx.stroke();
    rec.raf = requestAnimationFrame(drawWaveLive);
  }

  function setRecording(on) {
    $("#recorder").classList.toggle("recording", on);
    $("#recBtn").setAttribute("aria-label", on ? "Stop recording" : "Start recording");
  }

  async function startRecording() {
    hideError();
    if (!navigator.mediaDevices?.getUserMedia) {
      return showError("Your browser can't access the microphone here. Open the app at <code>http://localhost:5000</code> in Chrome or Edge, or upload a recording instead.");
    }
    try {
      rec.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: true, autoGainControl: false } });
    } catch {
      return showError("Microphone access was blocked. Allow the microphone in your browser's address bar, or upload a recording instead.");
    }
    rec.ctx = new (window.AudioContext || window.webkitAudioContext)();
    rec.sr = rec.ctx.sampleRate;
    const src = rec.ctx.createMediaStreamSource(rec.stream);
    rec.analyser = rec.ctx.createAnalyser();
    rec.analyser.fftSize = 2048;
    rec.proc = rec.ctx.createScriptProcessor(4096, 1, 1);
    rec.chunks = [];
    rec.proc.onaudioprocess = (e) => rec.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    src.connect(rec.analyser);
    src.connect(rec.proc);
    rec.proc.connect(rec.ctx.destination);           // required for the processor to run (outputs silence)

    rec.start = performance.now();
    setRecording(true);
    sizeCanvas();
    drawWaveLive();
    $("#recStatus").textContent = "Listening… tap again when you're done";
    rec.timer = setInterval(() => {
      const s = (performance.now() - rec.start) / 1000;
      $("#recTime").textContent = fmtTime(s);
      $("#recProgress").style.width = `${Math.min(100, (s / MAX_S) * 100)}%`;
      if (s < MIN_S) $("#recStatus").textContent = "Listening… keep going";
      else $("#recStatus").textContent = "Listening… tap again when you're done";
      if (s >= MAX_S) stopRecording();
    }, 100);
  }

  function teardown() {
    clearInterval(rec.timer);
    cancelAnimationFrame(rec.raf);
    rec.proc && (rec.proc.onaudioprocess = null);
    rec.stream?.getTracks().forEach((t) => t.stop());
    rec.ctx?.close();
    rec.stream = rec.ctx = rec.proc = null;
    setRecording(false);
    $("#recProgress").style.width = "0";
    $("#recTime").textContent = "0:00";
    drawWaveIdle();
  }

  function stopRecording() {
    const secs = (performance.now() - rec.start) / 1000;
    const chunks = rec.chunks, sr = rec.sr;
    teardown();
    if (secs < MIN_S) {
      $("#recStatus").textContent = "Tap to start · speak for 5–15 seconds";
      return showError(`That was a bit short — please speak for at least ${MIN_S} seconds so there's enough to listen to.`);
    }
    const len = chunks.reduce((a, c) => a + c.length, 0);
    const samples = new Float32Array(len);
    let off = 0;
    for (const c of chunks) { samples.set(c, off); off += c.length; }
    analyse(encodeWav(samples, sr));
  }

  function encodeWav(samples, sr) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
    str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  $("#recBtn").addEventListener("click", () => (rec.stream ? stopRecording() : startRecording()));

  $("#fileInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    hideError();
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const audio = await ctx.decodeAudioData(await file.arrayBuffer());
      const n = Math.min(audio.length, audio.sampleRate * 30);
      const mono = new Float32Array(n);
      for (let ch = 0; ch < audio.numberOfChannels; ch++) {
        const d = audio.getChannelData(ch);
        for (let i = 0; i < n; i++) mono[i] += d[i] / audio.numberOfChannels;
      }
      const sr = audio.sampleRate;
      ctx.close();
      analyse(encodeWav(mono, sr));
    } catch {
      showError("That file couldn't be read as audio. Try a .wav, .mp3 or .m4a file.");
    }
  });

  // ------------------------------------------------------------------
  // Analyse + show result
  // ------------------------------------------------------------------
  let lastResult = null;

  function showError(html) {
    const b = $("#errorBox");
    b.innerHTML = html;
    b.hidden = false;
  }
  function hideError() { $("#errorBox").hidden = true; }

  async function analyse(blob) {
    if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
    lastBlobUrl = URL.createObjectURL(blob);
    $("#result").hidden = true;
    $("#analysing").hidden = false;
    $("#recBtn").disabled = true;
    $("#recStatus").textContent = "Thank you. Give me a moment…";
    $("#analysing").scrollIntoView({ behavior: "smooth", block: "center" });

    const form = new FormData();
    form.append("audio", blob, "checkin.wav");
    if (selectedModel) form.append("model", selectedModel);
    try {
      const r = await fetch("/api/analyze", { method: "POST", body: form });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.error || `The server returned an error (${r.status}).`);
      lastResult = { ...j, selfMood, ts: Date.now() };
      renderResult(lastResult);
    } catch (err) {
      const offline = err instanceof TypeError;
      showError(offline
        ? "Can't reach the analysis server. Start it from the project folder with <code>python webapp/server.py</code>, then reload this page."
        : esc(err.message));
    } finally {
      $("#analysing").hidden = true;
      $("#recBtn").disabled = false;
      $("#recStatus").textContent = "Tap to start · speak for 5–15 seconds";
    }
  }

  function describeMetrics(m) {
    const out = [];
    if (m.pitch_hz != null) out.push(["Pitch", `${Math.round(m.pitch_hz)} Hz`, "your average speaking pitch"]);
    if (m.pitch_variation != null) {
      const v = m.pitch_variation;
      out.push(["Expressiveness", v < 1.5 ? "Flat" : v < 3 ? "Natural" : "Lively",
        v < 1.5 ? "little pitch movement — can come with tiredness" : v < 3 ? "a normal amount of pitch movement" : "lots of rise and fall in your voice"]);
    }
    if (m.pace != null) {
      const p = m.pace;
      out.push(["Pace", p < 2.5 ? "Unhurried" : p < 4.5 ? "Steady" : "Quick", `${p.toFixed(1)} sound onsets per second`]);
    }
    if (m.pause_ratio != null) {
      const pr = m.pause_ratio;
      out.push(["Pauses", pct(pr), pr < 0.2 ? "you spoke almost continuously" : pr < 0.45 ? "a natural number of pauses" : "lots of pauses or hesitation"]);
    }
    return out;
  }

  function renderResult(res) {
    const e = res.emotion;
    const info = EMO[e] || { label: e, tone: e, headline: `Your voice sounds ${e}.`, message: "", score: 50 };
    const tone = emoColor(e);
    const hero = $(".result-hero");
    hero.style.setProperty("--tone", tone);
    $("#toneChip").style.setProperty("--tone", tone);
    $("#toneChip").innerHTML = `<span class="dot" style="background:${tone}"></span>${esc(info.tone)}`;
    $("#resHeadline").textContent = info.headline;
    $("#resMessage").textContent = info.message;

    const conf = res.confidence;
    $("#gaugeValue").textContent = pct(conf);
    $("#gaugeFill").style.strokeDashoffset = 326.7;
    requestAnimationFrame(() => requestAnimationFrame(() => { $("#gaugeFill").style.strokeDashoffset = 326.7 * (1 - conf); }));

    const sc = $("#selfCompare");
    if (res.selfMood) {
      const said = SELF_MOODS.find((m) => m.id === res.selfMood)?.label.toLowerCase();
      sc.hidden = false;
      sc.textContent = res.selfMood === e
        ? `You said you feel ${said}, and your voice agrees.`
        : `You said you feel ${said}; your voice sounded ${info.label.toLowerCase()}. Both can be true — feelings are often mixed.`;
    } else sc.hidden = true;
    if (conf < 0.4) {
      sc.hidden = false;
      sc.textContent = (sc.textContent ? sc.textContent + " " : "") + "The model wasn't very sure this time, so take this as a light suggestion.";
    }

    $("#playBtn").hidden = !lastBlobUrl;

    // recommendations
    $("#recs").innerHTML = (RECS[e] || RECS.neutral).map((r, i) => `
      <article class="rec">
        <div class="rec-icon" aria-hidden="true">${ICON[r.icon] || ICON.heart}</div>
        <h4>${esc(r.title)}</h4>
        <p>${esc(r.text)}</p>
        ${r.action ? `<button class="btn ghost small" data-rec="${i}">${esc(r.action.label)}</button>` : ""}
      </article>`).join("");

    // probability bars
    const probs = Object.entries(res.probabilities).sort((a, b) => b[1] - a[1]);
    $("#bars").innerHTML = probs.map(([k, p], i) => `
      <div class="bar-row${i === 0 ? " top" : ""}">
        <span class="name"><span class="dot" style="background:${emoColor(k)}"></span>${esc(EMO[k]?.label || k)}</span>
        <div class="bar-track"><div class="bar-fill" data-w="${(p * 100).toFixed(1)}" style="background:${emoColor(k)}"></div></div>
        <span class="pct">${pct(p)}</span>
      </div>`).join("");
    requestAnimationFrame(() => requestAnimationFrame(() => $$("#bars .bar-fill").forEach((b) => { b.style.width = `${b.dataset.w}%`; })));

    // metrics
    $("#metrics").innerHTML = describeMetrics(res.metrics).map(([k, v, d]) =>
      `<div class="metric"><div class="k">${k}</div><div class="v">${esc(v)}</div><div class="d">${esc(d)}</div></div>`).join("");

    // timeline
    const tl = res.timeline || [];
    $("#timeline").innerHTML = tl.map((w) =>
      `<span style="background:${emoColor(w.emotion)}" title="${w.start.toFixed(1)}s: ${esc(EMO[w.emotion]?.label || w.emotion)} (${pct(w.confidence)})"></span>`).join("");
    const kinds = [...new Set(tl.map((w) => w.emotion))];
    $("#timelineNote").textContent = tl.length <= 1
      ? "Speak for longer than 4 seconds to see how your tone changes over time."
      : kinds.length === 1
        ? `Your tone stayed ${(EMO[kinds[0]]?.label || kinds[0]).toLowerCase()} across all ${tl.length} parts of the recording.`
        : `Your tone shifted between ${kinds.map((k) => (EMO[k]?.label || k).toLowerCase()).join(", ")} across ${tl.length} parts of the recording.`;

    // pattern of low check-ins
    const recent = journal().slice(-4).map((x) => x.emotion).concat(e);
    const lows = recent.filter((x) => LOW.has(x)).length;
    const pb = $("#patternBox");
    if (recent.length >= 3 && lows >= 3) {
      pb.hidden = false;
      pb.innerHTML = `<strong>A gentle note:</strong> several of your recent check-ins have sounded low or tense. That happens to everyone sometimes — but you don't have to work through it alone. <a href="#support">Talking to someone</a> can really help.`;
    } else pb.hidden = true;

    $("#noteInput").value = "";
    $("#saveBtn").disabled = false;
    $("#saveBtn").textContent = "Save check-in";
    $("#result").hidden = false;
    $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  $("#recs").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-rec]");
    if (!b || !lastResult) return;
    const a = (RECS[lastResult.emotion] || RECS.neutral)[+b.dataset.rec].action;
    if (a.journal) {
      $("#noteInput").focus();
      $(".save-card").scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    if (a.breathe) selectTechnique(a.breathe);
    if (a.ground) setGround(0);
    location.hash = a.go;
  });

  $("#againBtn").addEventListener("click", () => {
    $("#result").hidden = true;
    hideError();
    promptIdx = (promptIdx + 1) % PROMPTS.length;
    showPrompt();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  $("#playBtn").addEventListener("click", () => {
    const p = $("#player");
    if (!lastBlobUrl) return;
    if (!p.paused) { p.pause(); return; }
    p.src = lastBlobUrl;
    p.play();
  });

  $("#saveBtn").addEventListener("click", () => {
    if (!lastResult) return;
    const entries = journal();
    entries.push({
      id: lastResult.ts, ts: lastResult.ts, emotion: lastResult.emotion, confidence: lastResult.confidence,
      selfMood: lastResult.selfMood, note: $("#noteInput").value.trim(), model: lastResult.model,
    });
    if (!store.set(JOURNAL_KEY, entries.slice(-500))) return toast("Couldn't save — your browser is blocking storage.");
    $("#saveBtn").disabled = true;
    $("#saveBtn").textContent = "Saved ✓";
    toast("Saved to your journal");
  });

  // ------------------------------------------------------------------
  // Breathe
  // ------------------------------------------------------------------
  let technique = "box", rounds = 4;
  const ROUND_OPTS = [[4, "≈1 min"], [8, "≈2 min"], [15, "≈4 min"]];
  $("#techniques").innerHTML = Object.entries(TECHNIQUES).map(([id, t]) =>
    `<button class="option" role="radio" aria-checked="${id === technique}" data-tech="${id}"><strong>${t.name}</strong><span>${t.desc}</span></button>`).join("");
  $("#rounds").innerHTML = ROUND_OPTS.map(([n, l]) =>
    `<button class="chip" role="radio" aria-checked="${n === rounds}" data-rounds="${n}">${n} rounds · ${l}</button>`).join("");
  function selectTechnique(id) {
    technique = id;
    $$("#techniques .option").forEach((o) => o.setAttribute("aria-checked", String(o.dataset.tech === id)));
    stopBreathing();
  }
  $("#techniques").addEventListener("click", (e) => { const b = e.target.closest("[data-tech]"); if (b) selectTechnique(b.dataset.tech); });
  $("#rounds").addEventListener("click", (e) => {
    const b = e.target.closest("[data-rounds]");
    if (!b) return;
    rounds = +b.dataset.rounds;
    $$("#rounds .chip").forEach((c) => c.setAttribute("aria-checked", String(+c.dataset.rounds === rounds)));
    stopBreathing();
  });

  const br = { on: false, timeouts: [], tick: 0 };
  const core = $(".orb-core");
  function stopBreathing() {
    br.on = false;
    br.timeouts.forEach(clearTimeout);
    clearInterval(br.tick);
    br.timeouts = [];
    core.style.transitionDuration = "1.2s";
    core.style.transform = "scale(.55)";
    $("#phase").textContent = "Ready when you are";
    $("#count").textContent = "";
    $("#roundInfo").textContent = "";
    $("#breatheBtn").textContent = "Begin";
  }
  function runBreathing() {
    const t = TECHNIQUES[technique];
    br.on = true;
    $("#breatheBtn").textContent = "Stop";
    let round = 1, phaseIdx = 0;
    const step = () => {
      if (!br.on) return;
      if (phaseIdx >= t.phases.length) { phaseIdx = 0; round += 1; }
      if (round > rounds) {
        stopBreathing();
        $("#phase").textContent = "Well done.";
        $("#count").textContent = "Notice how your body feels now.";
        return;
      }
      const [label, secs, kind] = t.phases[phaseIdx];
      $("#phase").textContent = label;
      $("#roundInfo").textContent = `Round ${round} of ${rounds}`;
      core.style.transitionDuration = `${secs}s`;
      if (kind === "in") core.style.transform = "scale(1)";
      if (kind === "out") core.style.transform = "scale(.55)";
      let left = secs;
      $("#count").textContent = left;
      clearInterval(br.tick);
      br.tick = setInterval(() => { left -= 1; if (left > 0) $("#count").textContent = left; }, 1000);
      phaseIdx += 1;
      br.timeouts.push(setTimeout(step, secs * 1000));
    };
    $("#phase").textContent = "Get comfortable…";
    $("#count").textContent = "";
    br.timeouts.push(setTimeout(step, 1500));
  }
  $("#breatheBtn").addEventListener("click", () => (br.on ? stopBreathing() : runBreathing()));

  // ------------------------------------------------------------------
  // Grounding
  // ------------------------------------------------------------------
  let gIdx = 0;
  $("#groundDots").innerHTML = GROUND.map(() => "<span></span>").join("");
  function setGround(i) {
    gIdx = Math.max(0, Math.min(GROUND.length, i));
    if (gIdx === GROUND.length) {
      $("#groundNum").textContent = "✓";
      $("#groundText").innerHTML = "You're here, in this moment. Take one slow breath before you carry on.";
      $("#groundNext").textContent = "Start over";
    } else {
      $("#groundNum").textContent = GROUND[gIdx][0];
      $("#groundText").innerHTML = GROUND[gIdx][1];
      $("#groundNext").textContent = "Next";
    }
    $$("#groundDots span").forEach((d, j) => d.classList.toggle("on", j < gIdx || gIdx === GROUND.length || j === gIdx));
    $("#groundBack").disabled = gIdx === 0;
  }
  $("#groundNext").addEventListener("click", () => setGround(gIdx >= GROUND.length ? 0 : gIdx + 1));
  $("#groundBack").addEventListener("click", () => setGround(gIdx - 1));
  setGround(0);

  // ------------------------------------------------------------------
  // Journal
  // ------------------------------------------------------------------
  function streak(entries) {
    const days = new Set(entries.map((x) => dayKey(new Date(x.ts))));
    const d = new Date();
    if (!days.has(dayKey(d))) d.setDate(d.getDate() - 1);     // streak still alive if you checked in yesterday
    let n = 0;
    while (days.has(dayKey(d))) { n += 1; d.setDate(d.getDate() - 1); }
    return n;
  }

  function renderJournal() {
    const entries = journal();
    const weekAgo = Date.now() - 7 * 864e5;
    const week = entries.filter((x) => x.ts >= weekAgo);
    const counts = {};
    entries.slice(-30).forEach((x) => { counts[x.emotion] = (counts[x.emotion] || 0) + 1; });
    const common = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    const avg = week.length ? Math.round(week.reduce((a, x) => a + (EMO[x.emotion]?.score ?? 50), 0) / week.length) : null;
    const s = streak(entries);

    $("#stats").innerHTML = [
      ["This week", week.length, week.length === 1 ? "check-in" : "check-ins"],
      ["Streak", s, s === 1 ? "day in a row" : "days in a row"],
      ["Most common", common ? `<span class="dot" style="background:${emoColor(common[0])};width:14px;height:14px"></span>${esc(EMO[common[0]]?.label || common[0])}` : "—", "across your last 30"],
      ["Weekly wellbeing", avg != null ? avg : "—", avg != null ? (avg >= 65 ? "mostly positive" : avg >= 45 ? "a mixed week" : "a tougher week — be gentle with yourself") : "no check-ins this week"],
    ].map(([k, v, d]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");

    renderChart(entries.slice(-14));

    $("#entries").innerHTML = entries.length
      ? entries.slice().reverse().map((x) => {
          const info = EMO[x.emotion] || { label: x.emotion, tone: "" };
          const self = x.selfMood ? ` · you said “${SELF_MOODS.find((m) => m.id === x.selfMood)?.label}”` : "";
          return `<li class="entry">
            <div class="swatch" style="background:color-mix(in srgb, ${emoColor(x.emotion)} 20%, transparent)"><span class="dot" style="background:${emoColor(x.emotion)}"></span></div>
            <div>
              <div class="title">${esc(info.label)} <span class="muted small">· ${esc(info.tone)} · ${pct(x.confidence)}</span></div>
              <div class="meta">${new Date(x.ts).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}${esc(self)}</div>
              ${x.note ? `<div class="note">${esc(x.note)}</div>` : ""}
            </div>
            <button class="del" data-del="${x.id}" aria-label="Delete entry"><svg viewBox="0 0 24 24"><path d="M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13"/></svg></button>
          </li>`;
        }).join("")
      : `<li class="empty"><strong>No check-ins yet</strong>Record your first check-in and save it — your journal will start here.</li>`;
  }

  function renderChart(entries) {
    const el = $("#chart");
    if (entries.length < 2) {
      el.innerHTML = `<div class="empty"><strong>Your trend will appear here</strong>Save at least two check-ins to see how your wellbeing changes.</div>`;
      return;
    }
    const W = 720, H = 220, P = { l: 36, r: 16, t: 16, b: 30 };
    const xs = (i) => P.l + (i * (W - P.l - P.r)) / (entries.length - 1);
    const ys = (v) => P.t + (1 - v / 100) * (H - P.t - P.b);
    const pts = entries.map((x, i) => [xs(i), ys(EMO[x.emotion]?.score ?? 50)]);
    const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
    const area = `${line}L${pts[pts.length - 1][0]},${H - P.b}L${pts[0][0]},${H - P.b}Z`;
    const grid = [0, 50, 100].map((v) => `<line class="grid-line" x1="${P.l}" x2="${W - P.r}" y1="${ys(v)}" y2="${ys(v)}"/><text class="axis-label" x="${P.l - 8}" y="${ys(v) + 4}" text-anchor="end">${v}</text>`).join("");
    const labels = entries.map((x, i) => (entries.length <= 7 || i % 2 === 0 || i === entries.length - 1)
      ? `<text class="axis-label" x="${xs(i)}" y="${H - 8}" text-anchor="middle">${new Date(x.ts).toLocaleDateString(undefined, { day: "numeric", month: "short" })}</text>` : "").join("");
    const dots = entries.map((x, i) => `<circle class="pt" cx="${pts[i][0]}" cy="${pts[i][1]}" r="6" fill="${emoColor(x.emotion)}"><title>${esc(EMO[x.emotion]?.label || x.emotion)} · ${new Date(x.ts).toLocaleString()}</title></circle>`).join("");
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Wellbeing score of your last ${entries.length} check-ins">
      <defs><linearGradient id="areaGrad" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="var(--accent)" stop-opacity=".25"/><stop offset="1" stop-color="var(--accent)" stop-opacity="0"/></linearGradient></defs>
      ${grid}<path class="area" d="${area}"/><path class="line" d="${line}"/>${dots}${labels}</svg>`;
  }

  $("#entries").addEventListener("click", (e) => {
    const b = e.target.closest("[data-del]");
    if (!b) return;
    store.set(JOURNAL_KEY, journal().filter((x) => String(x.id) !== b.dataset.del));
    renderJournal();
    toast("Entry deleted");
  });
  $("#clearBtn").addEventListener("click", () => {
    if (!journal().length) return;
    if (!confirm("Delete all journal entries from this browser? This can't be undone.")) return;
    store.set(JOURNAL_KEY, []);
    renderJournal();
    toast("Journal cleared");
  });
  $("#exportBtn").addEventListener("click", () => {
    const entries = journal();
    if (!entries.length) return toast("Nothing to export yet");
    const q = (s) => `"${String(s ?? "").replace(/"/g, '""')}"`;
    const rows = [["date", "voice_tone", "confidence", "self_reported", "note", "model"]]
      .concat(entries.map((x) => [new Date(x.ts).toISOString(), x.emotion, x.confidence, x.selfMood || "", x.note, x.model]));
    const blob = new Blob([rows.map((r) => r.map(q).join(",")).join("\n")], { type: "text/csv" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "feelspace-journal.csv" });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  // ------------------------------------------------------------------
  // Start
  // ------------------------------------------------------------------
  window.addEventListener("resize", () => { if (!rec.stream) drawWaveIdle(); });
  route();
  drawWaveIdle();
  loadModels();
})();
