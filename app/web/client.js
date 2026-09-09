// Mic -> AssemblyAI -> speaker, in the browser.
//
// The two AudioWorklet processors below are taken from AssemblyAI's starter
// client (voice-agent-starter/deployment/browser/app.js) with comments added.
// They are commodity audio plumbing — resampling and buffering — and writing
// them again from scratch would land on the same code. Everything after them,
// the session layer, is ours, because that is the part Brainstormer changes.

const $ = (id) => document.getElementById(id)

// The rate the API speaks, both directions. A browser may quietly ignore the
// rate an AudioContext asks for, so both worklets resample rather than trust it.
const WIRE_RATE = 24_000

// --- borrowed: capture ------------------------------------------------------
// Microphone float samples -> 16-bit PCM at 24 kHz, posted to the main thread.
// Scratch buffers are allocated once and reused: allocating on the audio thread
// causes audible glitches.
const CAPTURE_WORKLET = `
  class CaptureProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this._ratio = sampleRate / ${WIRE_RATE};
      this._pos = 0;
      this._prev = 0;
      this._src = null;
      this._out = null;
    }
    _toPcm(samples, len) {
      const pcm = new Int16Array(len);
      for (let i = 0; i < len; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      return pcm;
    }
    process(inputs) {
      const ch = inputs[0]?.[0];
      if (!ch) return true;
      if (this._ratio === 1) {
        const pcm = this._toPcm(ch, ch.length);
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
        return true;
      }
      const n = ch.length;
      if (!this._src || this._src.length < n + 1) {
        this._src = new Float32Array(n + 1);
        this._out = new Float32Array(Math.ceil((n + 1) / this._ratio) + 2);
      }
      const src = this._src;
      const out = this._out;
      src[0] = this._prev;
      src.set(ch, 1);
      let outLen = 0;
      let pos = this._pos;
      while (pos < n) {
        const i = Math.floor(pos);
        const frac = pos - i;
        out[outLen++] = src[i] + (src[i + 1] - src[i]) * frac;
        pos += this._ratio;
      }
      this._pos = pos - n;
      this._prev = ch[n - 1];
      if (outLen) {
        const pcm = this._toPcm(out, outLen);
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
      }
      return true;
    }
  }
  registerProcessor('capture', CaptureProcessor);
`

// --- borrowed: playback -----------------------------------------------------
// A ring buffer rather than one AudioBufferSource per chunk, which drifts and
// clicks under network jitter. Posting 'stop' empties it instantly, which is
// how barge-in cuts the agent off mid-word.
const PLAYBACK_WORKLET = `
  class PlaybackProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this._ring = new Float32Array(sampleRate * 30);
      this._writePos = 0;
      this._readPos = 0;
      this._available = 0;
      this._step = ${WIRE_RATE} / sampleRate;
      this._rsPos = 0;
      this._rsPrev = 0;
      // After the buffer runs dry the speaker sits at zero, so interpolating
      // from the pre-gap sample would click. Reset instead.
      this._drained = false;
      this.port.onmessage = (e) => {
        if (e.data === 'stop') {
          this._writePos = this._readPos = this._available = 0;
          this._rsPos = this._rsPrev = 0;
          return;
        }
        const int16 = new Int16Array(e.data);
        // int16[-1] would make _rsPrev NaN and silence the ring permanently.
        if (!int16.length) return;
        if (this._drained) {
          this._rsPrev = 0;
          this._rsPos = 0;
          this._drained = false;
        }
        if (this._step === 1) {
          for (let i = 0; i < int16.length; i++) this._push(int16[i] / 32768);
          return;
        }
        const n = int16.length;
        let pos = this._rsPos;
        while (pos < n) {
          const i = Math.floor(pos);
          const frac = pos - i;
          const a = i === 0 ? this._rsPrev : int16[i - 1] / 32768;
          const b = int16[i] / 32768;
          this._push(a + (b - a) * frac);
          pos += this._step;
        }
        this._rsPos = pos - n;
        this._rsPrev = int16[n - 1] / 32768;
      };
    }
    _push(v) {
      if (this._available < this._ring.length) {
        this._ring[this._writePos] = v;
        this._writePos = (this._writePos + 1) % this._ring.length;
        this._available++;
      }
    }
    process(inputs, outputs) {
      const output = outputs[0];
      const out = output[0];
      const cap = this._ring.length;
      for (let i = 0; i < out.length; i++) {
        if (this._available > 0) {
          out[i] = this._ring[this._readPos];
          this._readPos = (this._readPos + 1) % cap;
          this._available--;
        } else {
          out[i] = 0;
          this._drained = true;
        }
      }
      // Mono source, possibly stereo sink.
      for (let ch = 1; ch < output.length; ch++) output[ch].set(out);
      return true;
    }
  }
  registerProcessor('playback', PlaybackProcessor);
`

async function addWorklet(ctx, code, name) {
  const url = URL.createObjectURL(new Blob([code], { type: 'application/javascript' }))
  try {
    await ctx.audioWorklet.addModule(url)
  } finally {
    URL.revokeObjectURL(url)
  }
  return new AudioWorkletNode(ctx, name)
}

// --- ours: the session ------------------------------------------------------

let ws, captureCtx, playbackCtx, playback, mic, startedAt, timer

// The mode the buttons are set to, and the one actually loaded on the socket.
// They differ between clicking a mode before a call starts and the socket
// becoming ready.
let selectedMode = null
let liveMode = null
let modesById = {}

$('btn').onclick = () => (ws?.readyState <= 1 ? hangUp() : call())

// --- the instrument ---------------------------------------------------------
//
// One canvas, repainted every frame from three inputs: which state the session
// is in, how loud the live side of the conversation actually is, and how far
// through the silence window we are.
//
// The aura breathes on REAL audio rather than on the state name. An AnalyserNode
// taps the microphone on the way up and the playback worklet on the way down,
// so what you see moving is the waveform that is actually moving, in whichever
// direction is currently talking.
//
// The ring is the part that earns its place. `min_silence` is how long the agent
// waits after you stop before it takes its turn — 300 ms in gym, 1200 ms in deep
// work. It is the most consequential number in a profile and it is normally
// completely invisible. Here it is the arc: it starts closing on
// `input.speech.stopped`, which is the server's own turn-detection signal and
// not a guess of ours, and completes at min_silence, which is when the agent
// speaks. Swap mode mid-call and the ring visibly changes temperament.
const aura = (() => {
  const canvas = $('aura')
  const paint = canvas.getContext('2d')

  let state = 'idle'
  let hue = 190
  let minSilence = 0
  let waitStart = 0
  let level = 0          // smoothed amplitude of the live side, 0..1
  let micTap = null      // AnalyserNode on the microphone
  let agentTap = null    // AnalyserNode on the playback worklet
  let samples = null
  let drift = 0

  // What the readout says in each state. One place decides it, so the words on
  // screen can never disagree with what the canvas is drawing.
  const SAYS = {
    idle: 'not listening',
    connecting: 'opening the socket',
    listening: 'listening',
    waiting: 'waiting for you to finish',
    speaking: 'speaking',
    error: 'stopped',
  }

  // RMS across the analyser's current window. Speech sits low in the 0..1 range
  // so it is scaled up; without that the aura barely moves at normal talking
  // volume.
  function amplitude(tap) {
    if (!tap) return 0
    samples ??= new Uint8Array(tap.fftSize)
    tap.getByteTimeDomainData(samples)
    let sum = 0
    for (const sample of samples) {
      const centred = (sample - 128) / 128
      sum += centred * centred
    }
    return Math.min(1, Math.sqrt(sum / samples.length) * 4)
  }

  // The canvas is laid out by CSS; this keeps its pixel buffer in step with the
  // box it actually occupies, capped at 2x so a retina screen does not pay for
  // four times the fill rate.
  function fit() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2)
    const box = canvas.getBoundingClientRect()
    const width = Math.round(box.width * ratio)
    const height = Math.round(box.height * ratio)
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }
  }

  function blob(x, y, radius, alpha) {
    const fill = paint.createRadialGradient(x, y, 0, x, y, radius)
    fill.addColorStop(0, `hsl(${hue} 92% 68% / ${alpha})`)
    fill.addColorStop(0.45, `hsl(${hue} 88% 56% / ${alpha * 0.4})`)
    fill.addColorStop(1, `hsl(${hue} 85% 50% / 0)`)
    paint.fillStyle = fill
    paint.beginPath()
    paint.arc(x, y, radius, 0, Math.PI * 2)
    paint.fill()
  }

  function frame() {
    requestAnimationFrame(frame)
    fit()

    const width = canvas.width
    const height = canvas.height
    const cx = width / 2
    const cy = height / 2
    const unit = Math.min(width, height) / 2

    const talking = state === 'speaking' ? amplitude(agentTap)
      : state === 'listening' || state === 'waiting' ? amplitude(micTap)
      : 0
    level += (talking - level) * 0.18
    drift += 0.0035 + level * 0.012

    paint.clearRect(0, 0, width, height)

    // Three offset lobes, rotating at slightly different rates so the shape
    // never repeats exactly. Added rather than painted over each other, which
    // is what gives the overlaps their brightness.
    const idle = state === 'idle' || state === 'error'
    const swell = (idle ? 0.24 : 0.28) + level * 0.16
    const glow = idle ? 0.26 : 0.42 + level * 0.34

    // The lobes orbit WIDE rather than sitting on the centre, so the light
    // gathers into a corona and leaves the middle dark. That is not styling:
    // the readout sits in that hole, and with the lobes centred the glow washed
    // out the leading digit of a four-digit figure. Breathing outward also
    // shows amplitude better than a disc growing in place.
    paint.globalCompositeOperation = 'lighter'
    for (let i = 0; i < 3; i++) {
      const angle = drift * (1 + i * 0.35) + (i * Math.PI * 2) / 3
      const offset = unit * (0.24 + level * 0.10)
      blob(cx + Math.cos(angle) * offset,
           cy + Math.sin(angle) * offset,
           unit * swell * (1 + i * 0.12),
           glow / (1 + i * 0.55))
    }
    paint.globalCompositeOperation = 'source-over'

    // Punch the centre back out to the page colour. The lobes orbit, so at some
    // phases one of them crosses the middle and washes the figure underneath —
    // this guarantees the readout always sits on clean ground rather than
    // depending on where the rotation happens to be.
    const hole = paint.createRadialGradient(cx, cy, 0, cx, cy, unit * 0.40)
    hole.addColorStop(0, '#0c1319')
    hole.addColorStop(0.6, 'rgba(12, 19, 25, 0.82)')
    hole.addColorStop(1, 'rgba(12, 19, 25, 0)')
    paint.fillStyle = hole
    paint.fillRect(cx - unit * 0.4, cy - unit * 0.4, unit * 0.8, unit * 0.8)

    // The calibration ring, always drawn so the instrument has an edge even at
    // rest.
    const ring = unit * 0.66
    paint.lineWidth = Math.max(1, unit * 0.008)
    paint.strokeStyle = '#1e2c36'
    paint.beginPath()
    paint.arc(cx, cy, ring, 0, Math.PI * 2)
    paint.stroke()

    // The silence window, closing. Only while waiting — the rest of the time
    // there is no window open and an arc would be decoration.
    let elapsed = 0
    if (state === 'waiting' && minSilence > 0) {
      elapsed = Math.min(performance.now() - waitStart, minSilence)
      const progress = elapsed / minSilence
      paint.lineWidth = Math.max(2, unit * 0.02)
      paint.lineCap = 'round'
      paint.strokeStyle = `hsl(${hue} 85% 62%)`
      paint.beginPath()
      paint.arc(cx, cy, ring, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2)
      paint.stroke()
    }

    $('doing').textContent = SAYS[state] ?? state
    // While the window is closing, the figure counts toward the threshold; the
    // rest of the time it shows the threshold itself. Same number either way —
    // the parameter this mode chose.
    $('figure').innerHTML = minSilence
      ? `${state === 'waiting' ? Math.round(elapsed) : minSilence}<span class="unit"> ms</span>`
      : '&mdash;'
  }

  requestAnimationFrame(frame)

  return {
    set(next) {
      if (next === 'waiting') waitStart = performance.now()
      state = next
    },
    // Called on every profile swap. The page's whole palette hangs off --hue,
    // so this one line recolours the instrument, the pills, the agent's name in
    // the transcript and the switch markers together.
    tune(nextHue, silence) {
      hue = nextHue
      minSilence = silence
      document.documentElement.style.setProperty('--hue', nextHue)
    },
    // The analysers cannot exist before the call does, so they are handed over
    // at session start and dropped at teardown.
    listen(micNode, agentNode) {
      micTap = micNode
      agentTap = agentNode
    },
    stop() {
      micTap = agentTap = null
      level = 0
    },
  }
})()

// --- modes ------------------------------------------------------------------

async function loadModes() {
  const { modes } = await (await fetch('/profiles')).json()
  $('modes').replaceChildren()
  for (const mode of modes) {
    modesById[mode.id] = mode
    const button = document.createElement('button')
    button.className = 'mode'
    button.textContent = mode.name
    button.dataset.id = mode.id
    button.onclick = () => selectMode(mode.id)
    $('modes').append(button)
  }
  if (modes.length) selectMode(modes[0].id)
}

function selectMode(id) {
  selectedMode = id
  for (const button of $('modes').children) {
    button.classList.toggle('on', button.dataset.id === id)
  }
  // Recolour and re-scale the instrument immediately, so picking a mode shows
  // what it is before the call starts. Mid-call, applyMode does it again from
  // the assembled profile, which is the authoritative copy.
  const mode = modesById[id]
  if (mode) {
    aura.tune(mode.hue, mode.min_silence)
    $('live-mode').textContent = mode.name
  }
  // Mid-call, the switch happens now. Before a call, it is remembered and
  // applied the moment the session is ready.
  if (ws?.readyState === 1 && liveMode !== id) applyMode(id)
}

async function applyMode(id) {
  const profile = await (await fetch(`/profile?mode=${encodeURIComponent(id)}`)).json()
  if (profile.error) return fail(profile.error)

  // The entire swap: one session.update, on the socket that is already open.
  // No reconnect, no new session, nothing said so far is lost.
  send({ type: 'session.update', session: profile.session })
  liveMode = id

  const listen = profile.session.input
  const td = listen.turn_detection
  aura.tune(profile.hue, td.min_silence)
  $('live-mode').textContent = profile.name
  showSwitch(profile.name,
    `silence ${td.min_silence}–${td.max_silence}ms · ` +
    `barge-in delay ${td.interruption_delay}ms · ` +
    `${listen.transcription_mode} · ` +
    `${listen.keyterms.length} keyterms · ` +
    // The toolset is part of what a swap changes, so it belongs in the proof.
    `${profile.session.tools.length
        ? 'tools: ' + profile.session.tools.map((tool) => tool.name).join(', ')
        : 'no tools'} · ` +
    `prompt ${profile.budget.total} tok ` +
    `(l0 ${profile.budget.l0} + l1 ${profile.budget.l1} + l2 ${profile.budget.l2})`,
    profile.hue)
}

loadModes()

async function call() {
  $('btn').disabled = true
  setStatus('connecting')

  try {
    // The key never reaches this page; this token is valid for 60 seconds.
    const res = await fetch('/token')
    if (!res.ok) return fail('could not mint a token — check the API key')
    const { token } = await res.json()

    // Two contexts, created inside the click handler because Safari will not
    // start an AudioContext outside a user gesture.
    captureCtx = new AudioContext({ sampleRate: WIRE_RATE })
    playbackCtx = new AudioContext({ sampleRate: WIRE_RATE })
    await Promise.all([captureCtx.resume(), playbackCtx.resume()])

    playback = await addWorklet(playbackCtx, PLAYBACK_WORKLET, 'playback')
    // The instrument taps the agent's voice on its way to the speaker. An
    // AnalyserNode passes audio through untouched, so this changes nothing
    // about what is heard — it only gives the aura something real to move to.
    const agentTap = playbackCtx.createAnalyser()
    agentTap.fftSize = 512
    playback.connect(agentTap)
    agentTap.connect(playbackCtx.destination)

    mic = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        // Echo cancellation on, or the agent hears its own voice through the
        // speakers and interrupts itself forever.
        echoCancellation: true,
        // Both off: AssemblyAI does its own noise handling, and automatic gain
        // fights the voice-activity detector.
        noiseSuppression: false,
        autoGainControl: false,
      },
    })
    const capture = await addWorklet(captureCtx, CAPTURE_WORKLET, 'capture')
    const micSource = captureCtx.createMediaStreamSource(mic)
    micSource.connect(capture)
    // The same tap on the way up. Deliberately NOT connected onward to the
    // destination: this branch only measures, and wiring it to the speaker
    // would play your own voice back at you.
    const micTap = captureCtx.createAnalyser()
    micTap.fftSize = 512
    micSource.connect(micTap)
    aura.listen(micTap, agentTap)

    const url = new URL('wss://agents.assemblyai.com/v1/ws')
    url.searchParams.set('token', token)
    ws = new WebSocket(url)
    let ready = false

    // Audio goes up as base64 inside JSON, not as binary frames.
    capture.port.onmessage = ({ data }) => {
      if (!ready || ws.readyState !== 1) return
      const bytes = new Uint8Array(data)
      let binary = ''
      for (let i = 0; i < bytes.length; i += 0x8000) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000))
      }
      ws.send(JSON.stringify({ type: 'input.audio', audio: btoa(binary) }))
    }

    ws.onopen = () => {
      // The first message binds our stored agent, and `agent_id` must travel
      // ALONE — sent alongside any inline field it is rejected with
      // `agent_id_not_first`. Everything the agent needs (prompt, voice, model)
      // is already on the stored config.
      //
      // Step 2 is what happens after this: more session.update messages, this
      // time carrying inline fields, swapping the profile mid-call. The socket
      // stays open. Try it by hand right now from the console:
      //   brainstormer.update({ system_prompt: "Reply only in questions." })
      send({ type: 'session.update', session: { agent_id: window.AGENT_ID } })
    }

    ws.onmessage = ({ data }) => {
      const msg = JSON.parse(data)
      switch (msg.type) {
        case 'session.ready':
          ready = true
          startedAt = Date.now()
          timer = setInterval(tick, 1000)
          tick()
          setStatus('listening')
          $('btn').disabled = false
          $('btn').textContent = 'End call'
          $('btn').classList.add('live')
          // The stored agent's own prompt got us this far. Now load the real
          // profile — this is the first swap of every call.
          if (selectedMode) applyMode(selectedMode)
          break

        // The user started talking over the agent. Empty the playback buffer so
        // it stops immediately instead of finishing its sentence.
        case 'input.speech.started':
          playback?.port.postMessage('stop')
          setStatus('listening')
          break

        // Turn detection says the user has stopped. The silence window is now
        // open, and this is the server's own signal rather than something we
        // inferred from the microphone. The ring closes across min_silence from
        // here; when it completes, reply.started arrives.
        case 'input.speech.stopped':
          setStatus('waiting')
          break

        case 'reply.started':
          setStatus('speaking')
          break

        case 'reply.audio':
          playback?.port.postMessage(decodeAudio(msg.data), [])
          break

        case 'reply.done':
          if (msg.status === 'interrupted') playback?.port.postMessage('stop')
          setStatus('listening')
          break

        // Partial transcript: `text` is everything heard this turn so far, so
        // it replaces the line rather than appending to it.
        case 'transcript.user.delta':
          partial(msg.text)
          break

        case 'transcript.user':
          addLine('you', msg.text)
          break

        // Word-level, aligned to the audio as it plays — so the caption keeps
        // pace with the voice rather than landing all at once.
        case 'transcript.agent.delta':
          agentDelta(msg.reply_id, msg.delta)
          break

        // The whole reply, sent once its audio has been DELIVERED — which beats
        // the audio finishing playing, so deltas for this reply keep arriving
        // after this line prints. finishedReply stops them rebuilding the same
        // sentence underneath it.
        case 'transcript.agent':
          finishedReply = msg.reply_id ?? finishedReply
          dropAgentPartial()
          addLine('agent', msg.text)
          break

        // The agent wants a tool. It keeps talking while we work — the tool is
        // registered with execution_mode "interactive" — so this must not block
        // anything here; it fires and answers whenever it comes back.
        case 'tool.call':
          runTool(msg)
          break

        case 'session.error':
          fail(`${msg.code}: ${msg.message}`)
          break

        case 'session.ended':
          ws.close()
          break
      }
    }

    ws.onclose = () => { setStatus('idle'); reset() }
    ws.onerror = () => fail('connection failed')
  } catch (err) {
    fail(err.message)
  }
}

function send(message) {
  if (ws?.readyState === 1) ws.send(JSON.stringify(message))
}

function decodeAudio(base64) {
  const raw = atob(base64)
  const bytes = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i)
  return bytes.buffer
}

function hangUp() {
  // Close politely so the session record is finalised on AssemblyAI's side —
  // that stored session is what the evolver reads later.
  if (ws?.readyState === 1) {
    send({ type: 'session.end' })
    const socket = ws
    setTimeout(() => { if (socket.readyState === 1) socket.close() }, 2000)
  } else {
    ws?.close()
  }
  teardown()
  setStatus('idle')
}

function teardown() {
  playback?.port.postMessage('stop')
  mic?.getTracks().forEach((track) => track.stop())
  captureCtx?.close()
  playbackCtx?.close()
  captureCtx = playbackCtx = playback = mic = null
  aura.stop()
  reset()
}

function reset() {
  clearInterval(timer)
  dropPartial()
  dropAgentPartial()
  agentReply = finishedReply = null
  // The socket is gone, so no profile is loaded on it any more. The button
  // selection survives; the next call re-applies it at session.ready.
  liveMode = null
  $('btn').disabled = false
  $('btn').textContent = 'Start call'
  $('btn').classList.remove('live')
}

function fail(message) {
  setStatus('error', message)
  teardown()
}

function setStatus(state, detail) {
  $('status-text').textContent = detail || state
  $('status-text').classList.toggle('error', state === 'error')
  aura.set(state)
}

function tick() {
  const seconds = Math.floor((Date.now() - startedAt) / 1000)
  $('elapsed').textContent =
    `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
}

// --- transcript -------------------------------------------------------------

let partialEl = null       // the user's live line
let agentEl = null         // the agent's live line, built from deltas
let agentText = ''
let agentReply = null      // reply_id the live agent line belongs to
let finishedReply = null   // reply_id already printed in full

function clearEmpty() {
  $('transcript').querySelector('.empty')?.remove()
}

// Deltas arrive sometimes with a leading space and sometimes without, so add
// one only when neither side has it and the delta is not punctuation that
// attaches to the word before it.
const ATTACHES_LEFT = /^[.,!?;:%°)\]}…'"’”]/
const NO_SPACE_AFTER = /[([{$\-\/'"‘“]$/

function appendDelta(text, delta) {
  if (!delta) return text
  if (!text) return delta
  if (/^\s/.test(delta) || /\s$/.test(text)) return text + delta
  if (ATTACHES_LEFT.test(delta) || NO_SPACE_AFTER.test(text)) return text + delta
  return text + ' ' + delta
}

function agentDelta(replyId, delta) {
  if (replyId && replyId === finishedReply) return  // already printed in full
  if (replyId !== agentReply) {
    agentReply = replyId
    dropAgentPartial()
  }
  clearEmpty()
  agentText = appendDelta(agentText, delta)
  if (agentEl) {
    agentEl.querySelector('.said').textContent = agentText
  } else {
    agentEl = makeLine('agent', agentText, 'partial')
    $('transcript').append(agentEl)
  }
  scrollDown()
}

function dropAgentPartial() {
  agentEl?.remove()
  agentEl = null
  agentText = ''
}

// `hue` is stamped on the element rather than inherited, so a marker keeps the
// colour of the profile it recorded. Without it, switching to gym later would
// repaint every earlier 'profile loaded — Deep work' line amber, quietly
// rewriting the history the marker exists to preserve.
function showSwitch(name, detail, hue) {
  clearEmpty()
  const line = document.createElement('div')
  line.className = 'switch'
  if (hue !== undefined) line.style.setProperty('--hue', hue)
  line.textContent = `profile loaded — ${name}`
  const small = document.createElement('span')
  small.className = 'detail'
  small.textContent = detail
  line.append(small)
  $('transcript').append(line)
  scrollDown()
}

// --- tools -------------------------------------------------------------------

// `arguments` is a reserved word in a function signature, hence the rename.
//
// The result must be a STRING, not an object — tool.result carries JSON that
// has already been serialised — and it must echo the call_id it came with, or
// the agent cannot tell which of its calls was answered. Our server already
// hands back a serialised string, so it is passed through untouched.
//
// Sent the moment the tool returns. There is no waiting for reply.done and no
// timing dance: the agent is filling the gap with a transition phrase and picks
// this up whenever it lands.
async function runTool({ call_id, name, arguments: args }) {
  const started = performance.now()
  const line = showTool(name, args)

  let result
  try {
    const response = await fetch('/tool', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, arguments: args }),
    })
    if (!response.ok) throw new Error(response.status)
    result = (await response.json()).result
  } catch (err) {
    // The agent is mid-sentence waiting on this. Something it can read and be
    // honest about beats letting its 20-second timeout run out in silence.
    console.error('tool relay failed', err)
    result = JSON.stringify({ error: 'the tool could not be reached' })
  }

  annotateTool(line, performance.now() - started, result)
  send({ type: 'tool.result', call_id, result })
}

function showTool(name, args) {
  clearEmpty()
  const line = document.createElement('div')
  line.className = 'tool'
  line.textContent = `${name} · ${Object.values(args ?? {}).join(' ')}`
  const detail = document.createElement('span')
  detail.className = 'detail'
  detail.textContent = 'running…'
  line.append(detail)
  $('transcript').append(line)
  scrollDown()
  return line
}

// How long it took and how much it cost the context. The token figure is the
// point of the whole shaping step in tools.py, so it is worth being able to see
// it move on a real call rather than only in the module's own CLI.
function annotateTool(line, elapsedMs, result) {
  const parsed = JSON.parse(result)
  const detail = line.querySelector('.detail')
  detail.textContent = parsed.error
    ? `failed — ${parsed.error}`
    : `${Math.round(elapsedMs)}ms · ~${Math.round(result.length / 4)} tokens back` +
      (parsed.sources?.length ? ` · ${parsed.sources.length} sources` : '')
  scrollDown()
}

function makeLine(who, text, extra) {
  const line = document.createElement('div')
  line.className = `line ${who}${extra ? ' ' + extra : ''}`
  const label = document.createElement('span')
  label.className = 'who'
  label.textContent = who
  const said = document.createElement('span')
  said.className = 'said'
  said.textContent = text
  line.append(label, said)
  return line
}

function partial(text) {
  clearEmpty()
  if (partialEl) {
    partialEl.querySelector('.said').textContent = text
  } else {
    partialEl = makeLine('you', text, 'partial')
    $('transcript').append(partialEl)
  }
  scrollDown()
}

function dropPartial() {
  partialEl?.remove()
  partialEl = null
}

function addLine(who, text) {
  clearEmpty()
  if (who === 'you') dropPartial()
  $('transcript').append(makeLine(who, text))
  scrollDown()
}

function scrollDown() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
}

// Exposed so the profile swap can be tried by hand before step 2 builds it:
//   brainstormer.update({ system_prompt: "Answer only in questions." })
//   brainstormer.update({ input: { turn_detection: { max_silence: 4000 } } })
window.brainstormer = {
  update: (session) => send({ type: 'session.update', session }),
}
