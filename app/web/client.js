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

$('btn').onclick = () => (ws?.readyState <= 1 ? hangUp() : call())

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
    playback.connect(playbackCtx.destination)

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
    captureCtx.createMediaStreamSource(mic).connect(capture)

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
          break

        // The user started talking over the agent. Empty the playback buffer so
        // it stops immediately instead of finishing its sentence.
        case 'input.speech.started':
          playback?.port.postMessage('stop')
          setStatus('listening')
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

        // The agent's text also arrives as deltas, but the finalised message
        // carries the whole reply. Step 1 only prints the final one; stitching
        // deltas is fiddly and buys nothing yet.
        case 'transcript.agent':
          addLine('agent', msg.text)
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
  reset()
}

function reset() {
  clearInterval(timer)
  dropPartial()
  $('btn').disabled = false
  $('btn').textContent = 'Start call'
  $('btn').classList.remove('live')
}

function fail(message) {
  setStatus('error', message)
  teardown()
}

function setStatus(state, detail) {
  $('status').className = 'status ' + state
  $('status-text').textContent = detail || state
}

function tick() {
  const seconds = Math.floor((Date.now() - startedAt) / 1000)
  $('elapsed').textContent =
    `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
}

// --- transcript -------------------------------------------------------------

let partialEl = null

function clearEmpty() {
  $('transcript').querySelector('.empty')?.remove()
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
