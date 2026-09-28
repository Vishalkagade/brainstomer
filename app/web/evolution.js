// The evolution page: pick a mode, pick a version, see what changed and why.
// Reads /versions once, /changes per selection, posts /promote on a confirmed click.

const $ = (id) => document.getElementById(id)
const el = (tag, className, text) => {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

let modes = []          // [{id, name, status, hue, versions: [...]}]
let canPromote = true
let modeId = null
let versionId = null    // the version being looked at
let against = 'previous' // previous | live

// Settings keys in the words the pitch uses, with a unit where there is one.
const LABELS = {
  'turn_detection.min_silence': ['shortest wait', 'ms'],
  'turn_detection.max_silence': ['longest wait', 'ms'],
  'turn_detection.interruption_delay': ['interruption delay', 'ms'],
  'turn_detection.interrupt_response': ['can be interrupted'],
  transcription_mode: ['transcription'],
  keyterms: ['words to expect'],
  tools: ['tools'],
  history_depth: ['memory depth', 'turns'],
  about: ['about'],
  name: ['name'],
  hue: ['colour'],
  model: ['model'],
  'evolve_after.sessions': ['evolve after', 'calls'],
  'evolve_after.turns': ['evolve after', 'turns'],
}

const mode = () => modes.find((m) => m.id === modeId)
const day = (iso) => new Date(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })

async function load() {
  let data
  try {
    const response = await fetch('/versions')
    if (!response.ok) throw new Error(`the server answered ${response.status}`)
    data = await response.json()
  } catch (err) {
    $('change').replaceChildren(el('p', 'note', `Could not load the versions: ${err.message}. Reload the page, or check that the server is running.`))
    return
  }
  modes = data.modes
  canPromote = data.can_promote
  const query = new URLSearchParams(location.search)
  modeId = modes.some((m) => m.id === query.get('mode')) ? query.get('mode') : modes[0]?.id
  const wanted = Number(query.get('v'))
  pickMode(modeId, mode()?.versions.find((v) => v.n === wanted)?.id)
}

// What a version is, in one word. `calls` is how many calls ran on it.
function standing(v) {
  if (v.live) return 'live'
  if (v.verdict === 'rolled_back') return 'rolledback'
  if (v.calls === 0 && v.source === 'evolver') return 'proposed'
  return v.calls ? 'earlier' : 'unused'
}

// The evidence's word on an evolver version: being judged, kept, or rolled back. '' for hand-made versions.
function verdictLine(v) {
  if (v.verdict_why) return v.verdict_why
  if (v.live && v.source === 'evolver' && v.parent_id) return 'went live by itself; the next calls decide if it stays'
  return ''
}

function pickMode(id, preferredVersion) {
  modeId = id
  const m = mode()
  document.documentElement.style.setProperty('--hue', m.hue)
  $('modes').replaceChildren(...modes.map((each) => {
    const live = each.versions.find((v) => v.live)
    const button = el('button', 'mode' + (each.id === id ? ' on' : ''), each.name)
    button.append(el('small', '', `v${live.n}`))
    button.onclick = () => pickMode(each.id)
    return button
  }))
  // default to the most interesting thing: a waiting proposal, else the live version
  const proposal = m.versions.find((v) => standing(v) === 'proposed')
  pickVersion(preferredVersion ?? proposal?.id ?? m.versions.find((v) => v.live).id)
}

function pickVersion(id) {
  versionId = id
  const v = mode().versions.find((each) => each.id === id)
  history.replaceState(null, '', `/evolution?mode=${modeId}&v=${v.n}`)
  drawTimeline()
  drawChange()
}

function drawTimeline() {
  const newestFirst = [...mode().versions].reverse()
  $('timeline').replaceChildren(...newestFirst.map((v) => {
    const state = standing(v)
    const node = el('li', `node ${state}` + (v.id === versionId ? ' on' : ''))
    const line = el('div')
    const tag = { live: 'live', proposed: 'proposed', rolledback: 'rolled back' }[state] || ''
    line.append(el('span', 'n', `v${v.n}`), el('span', 'tag', tag))
    const calls = v.calls ? ` · ran in ${v.calls} call${v.calls > 1 ? 's' : ''}` : ''
    node.append(line, el('div', 'when', `${day(v.created_at)} · ${v.source}${calls}`))
    if (verdictLine(v)) node.append(el('div', 'when verdict', verdictLine(v)))
    node.onclick = () => pickVersion(v.id)
    // reachable by keyboard too, not only by mouse
    node.tabIndex = 0
    node.setAttribute('role', 'button')
    node.setAttribute('aria-pressed', String(v.id === versionId))
    node.onkeydown = (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); pickVersion(v.id) } }
    if (!v.live && canPromote) node.append(promoteButton(v, state))
    return node
  }))
}

// Two clicks: the first arms it, the second sends it. Rollback is the same call with an older id.
function promoteButton(v, state) {
  const label = state === 'proposed' ? 'Promote' : 'Make live again'
  const button = el('button', '', label)
  button.onclick = async (event) => {
    event.stopPropagation()
    if (!button.classList.contains('armed')) {
      button.classList.add('armed')
      button.textContent = `Confirm: v${v.n} goes live`
      setTimeout(() => { button.classList.remove('armed'); button.textContent = label }, 4000)
      return
    }
    button.disabled = true
    const response = await fetch('/promote', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ version_id: v.id }),
    })
    const result = await response.json()
    toast(response.ok
      ? `${mode().name} is now on v${result.n}. Router fingerprint ${result.fingerprint_refreshed ? 'refreshed' : 'unchanged, no embedding spent'}.`
      : result.error)
    const keep = versionId
    const data = await (await fetch('/versions')).json()
    modes = data.modes
    pickMode(modeId, keep)
  }
  return button
}

async function drawChange() {
  const versions = mode().versions
  const v = versions.find((each) => each.id === versionId)
  const previous = versions.filter((each) => each.n < v.n).pop()
  const live = versions.find((each) => each.live)
  const base = against === 'live' && !v.live ? live : previous
  const box = $('change')

  const title = el('h3', 'title')
  if (base) title.append(`v${base.n}`, el('span', 'arrow', '→'), `v${v.n}`)
  else title.append(`v${v.n}`)
  const sub = el('p', 'sub', base ? 'compared with ' : 'where this mode started. ')
  if (base && !v.live && previous && previous.id !== live.id) {
    for (const choice of ['previous', 'live']) {
      const link = el('a', against === choice ? 'on' : '', choice === 'live' ? `live v${live.n}` : `previous v${previous.n}`)
      link.href = `#${choice}`  // a real link: focusable, works with Enter
      link.onclick = (event) => { event.preventDefault(); against = choice; drawChange() }
      sub.append(link, ' ')
    }
  } else if (base) sub.append(`v${base.n}`)

  const why = el('p', 'why', v.rationale.trim() || 'No reason was recorded.')
  box.replaceChildren(title, sub, el('h2', '', 'Why'), why)

  const data = await (await fetch(`/changes?old=${(base || v).id}&new=${v.id}`)).json()
  if (versionId !== v.id) return // a newer click won the race

  if (base) {
    box.append(el('h2', '', 'Listening and tools'))
    box.append(data.settings.length ? settingsRows(data.settings) : el('p', 'note', 'No setting changed.'))
  }
  box.append(el('h2', '', base ? 'Prompt' : 'Prompt as first written'))
  const touched = data.prompt.some((op) => op.op !== 'same')
  if (base && !touched) box.append(el('p', 'note', 'The prompt did not change. Only the settings above did.'))
  else box.append(...data.prompt.map(paragraph))
}

function settingsRows(rows) {
  const table = el('div', 'rows')
  for (const row of rows) {
    const [label, unit] = LABELS[row.key] || [row.key]
    const line = el('div', 'row')
    const value = el('div')
    if ('added' in row) {
      value.append(...row.removed.map((x) => el('span', 'chip del', x)), ...row.added.map((x) => el('span', 'chip add', x)))
    } else {
      const show = (x) => (x === null || x === undefined ? 'not set' : `${x}${unit && typeof x === 'number' ? ' ' + unit : ''}`)
      value.append(el('span', 'old', show(row.old)), el('span', 'to', '→'), el('span', '', show(row.new)))
      if (typeof row.old === 'number' && typeof row.new === 'number') {
        const delta = row.new - row.old
        value.append(el('span', 'delta', `${delta > 0 ? '+' : ''}${delta}`))
      }
    }
    line.append(el('div', 'k', label), value)
    table.append(line)
  }
  return table
}

function paragraph(op) {
  const whole = op.op !== 'same' && op.parts.every((part) => !part.hit) // rewritten outright, no word marks
  const p = el('p', `para ${op.op}` + (whole ? ' whole' : ''))
  for (const part of op.parts) p.append(part.hit ? el('mark', '', part.t) : part.t)
  return p
}

let toastTimer
function toast(text) {
  $('toast').textContent = text
  $('toast').classList.add('show')
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => $('toast').classList.remove('show'), 5000)
}

load()
