/* NetSim3D — interface wiring. */

import { Sim } from './viz.js';

const $ = (id) => document.getElementById(id);

const stage = $('stage');
const dropzone = $('dropzone');
const fileInput = $('file');
const statusEl = $('status');
const inspectTitle = $('inspect-title');
const inspectShape = $('inspect-shape');
const inspectNote = $('inspect-note');
const inspectStats = $('inspect-stats');
const verdictEl = $('verdict');
const verdictMain = $('verdict-main');
const verdictSub = $('verdict-sub');
const verdictList = $('verdict-list');
const verdictWarn = $('verdict-warn');
const scrub = $('scrub');
const playBtn = $('play');
const recordBtn = $('record');
const chapters = $('chapters');
const speedSel = $('speed');
const recenterBtn = $('recenter');
const glowSel = $('glow');

let sim = null;
let payload = null;
let recorder = null;
let chunks = [];

function setStatus(text, kind = 'info') {
  statusEl.textContent = text || '';
  statusEl.dataset.kind = kind;
  statusEl.classList.toggle('is-on', Boolean(text));
}

/* ---------- boot ---------- */

try {
  sim = new Sim(stage);
} catch (err) {
  setStatus('3D could not start. Your browser needs WebGL: ' + err.message, 'error');
}

if (sim) {
  sim.setGlow(Number(glowSel.value));
  sim.onSegment = (seg, k, t) => {
    scrub.value = String(Math.round((t / sim.duration) * 1000));
    paintInspector(seg);
    [...chapters.children].forEach((c) => c.classList.toggle('is-live', c.dataset.id === seg.id));
    if (payload && (seg.id === 'output' || seg.id === 'cam' || seg.id === 'outro')) {
      verdictEl.classList.add('is-on');
    } else {
      verdictEl.classList.remove('is-on');
    }
  };
  sim.onEnd = () => { playBtn.textContent = 'Replay'; playBtn.dataset.mode = 'replay'; };
  sim.onManual = () => { recenterBtn.classList.add('is-on'); };
}

async function checkModel() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    if (s.error) {
      setStatus('Model failed to load: ' + s.error, 'error');
    } else if (s.loading) {
      setStatus('Loading ' + s.model + '. First run downloads the weights, about 45 MB.', 'wait');
      setTimeout(checkModel, 1500);
    } else {
      setStatus('');
      $('model-name').textContent = s.model + (s.accuracy ? ` · ${s.accuracy.toFixed(1)}% top-1` : '');
      $('model-meta').textContent = `${s.classes} classes, ${s.vehicle_classes} vehicle types`;
    }
  } catch (err) {
    setStatus('Cannot reach the Python server. Is the run window still open?', 'error');
  }
}
checkModel();

/* ---------- upload ---------- */

function wireDrop(el) {
  el.addEventListener('dragover', (e) => { e.preventDefault(); el.classList.add('is-hot'); });
  el.addEventListener('dragleave', () => el.classList.remove('is-hot'));
  el.addEventListener('drop', (e) => {
    e.preventDefault();
    el.classList.remove('is-hot');
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) send(f);
  });
}
wireDrop(dropzone);
wireDrop(document.body);

dropzone.addEventListener('click', () => fileInput.click());
$('pick').addEventListener('click', () => fileInput.click());
$('newimg').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) send(fileInput.files[0]);
});

async function send(file) {
  if (!sim) return;
  setStatus('Running the forward pass in Python.', 'wait');
  dropzone.classList.add('is-busy');
  const body = new FormData();
  body.append('image', file);
  try {
    const r = await fetch('/api/analyze', { method: 'POST', body });
    const data = await r.json();
    if (!data.ok) throw new Error(data.error || 'unknown error');
    payload = data;
    setStatus('');
    dropzone.classList.remove('is-on');
    $('shell').classList.add('has-run');
    buildChapters(sim.build(data));
    paintVerdict(data);
    playBtn.textContent = 'Pause';
    playBtn.dataset.mode = 'pause';
    sim.replay();
  } catch (err) {
    setStatus(err.message, 'error');
  } finally {
    dropzone.classList.remove('is-busy');
  }
}

/* ---------- panels ---------- */

function buildChapters(segments) {
  chapters.innerHTML = '';
  segments.forEach((s, i) => {
    const b = document.createElement('button');
    b.className = 'chapter';
    b.dataset.id = s.id;
    b.textContent = s.title;
    b.addEventListener('click', () => {
      sim.seekSegment(i);
      sim.play();
      playBtn.textContent = 'Pause';
      playBtn.dataset.mode = 'pause';
    });
    chapters.appendChild(b);
  });
}

function paintInspector(seg) {
  if (!payload) return;
  const layer = payload.layers.find((l) => l.id === seg.layerId);
  if (seg.id === 'input') {
    inspectTitle.textContent = 'Your image';
    inspectShape.textContent = `${payload.input.size} x ${payload.input.size} x 3`;
    inspectNote.textContent = 'Resized and centre-cropped, then normalised with the ImageNet mean and standard deviation. This tensor is what the first convolution actually sees.';
    inspectStats.textContent = `uploaded at ${payload.input.original[0]} x ${payload.input.original[1]}`;
    return;
  }
  if (seg.id === 'cam') {
    inspectTitle.textContent = 'Where it looked';
    inspectShape.textContent = `${(payload.attribution || {}).method || 'attribution'}, stage 4`;
    inspectNote.textContent = 'The gradient of the winning class with respect to the last feature maps, averaged into a weight per map. Bright regions are the pixels that pushed the decision. If the heat sits off the object, the answer was luck.';
    inspectStats.textContent = '';
    return;
  }
  if (seg.id === 'outro') {
    inspectTitle.textContent = 'Whole network';
    inspectShape.textContent = `${payload.layers.length} stages captured`;
    inspectNote.textContent = 'Drag to orbit, scroll to zoom. Every tile you see is a real activation from your image, not a decoration.';
    inspectStats.textContent =
      `${payload.timing_ms.forward} ms on ${payload.device || 'cpu'}`
      + (payload.views > 1 ? `  ·  answer averaged over ${payload.views} views of your photo` : '');
    return;
  }
  if (!layer) return;
  if (layer.kind === 'output') {
    const win = layer.top[0];
    inspectTitle.textContent = layer.title;
    inspectShape.textContent = `${layer.shape[0]} classes`;
    inspectNote.textContent = layer.note;
    inspectStats.textContent =
      `winner ${win.label} at ${(win.p * 100).toFixed(1)}%  ·  ` +
      `runner-up ${layer.top[1] ? layer.top[1].label : '—'} at ` +
      `${layer.top[1] ? (layer.top[1].p * 100).toFixed(1) : '0'}%`;
    return;
  }
  inspectTitle.textContent = layer.title;
  inspectShape.textContent = layer.shape.length === 3
    ? `${layer.shape[0]} maps of ${layer.shape[1]} x ${layer.shape[2]}`
    : `${layer.shape[0]} values`;
  inspectNote.textContent = layer.note;
  if (layer.stats) {
    inspectStats.textContent =
      `showing ${layer.shown || 0} strongest of ${layer.shape[0]}  ·  ` +
      `mean ${layer.stats.mean}  ·  peak ${layer.stats.max}  ·  ` +
      `${Math.round(layer.stats.alive * 100)}% of units firing`;
  } else {
    inspectStats.textContent = '';
  }
}

function paintVerdict(data) {
  const v = data.verdict;
  const out = data.layers.find((l) => l.kind === 'output');
  // The server decides what level to answer at: a confident family beats a
  // shaky fine class, and it says so rather than pretending otherwise.
  verdictMain.textContent = v.label;
  verdictSub.textContent = v.detail || `${(v.p * 100).toFixed(1)}% confident`;
  verdictWarn.textContent = v.warning || '';
  verdictWarn.classList.toggle('is-on', Boolean(v.warning));
  verdictList.innerHTML = '';
  (out ? out.top : []).forEach((e, i) => {
    const row = document.createElement('div');
    row.className = 'vrow' + (i === 0 ? ' is-win' : '');
    const name = document.createElement('span');
    name.className = 'vname';
    name.textContent = e.label;
    const bar = document.createElement('span');
    bar.className = 'vbar';
    bar.style.setProperty('--p', `${Math.max(1, e.p * 100).toFixed(1)}%`);
    const pct = document.createElement('span');
    pct.className = 'vpct';
    pct.textContent = `${(e.p * 100).toFixed(1)}%`;
    row.append(name, bar, pct);
    verdictList.appendChild(row);
  });
}

/* ---------- playback ---------- */

playBtn.addEventListener('click', () => {
  if (!payload) return;
  if (playBtn.dataset.mode === 'replay') {
    sim.replay();
    playBtn.textContent = 'Pause';
    playBtn.dataset.mode = 'pause';
  } else if (sim.playing) {
    sim.pause();
    playBtn.textContent = 'Play';
    playBtn.dataset.mode = 'play';
  } else {
    sim.play();
    playBtn.textContent = 'Pause';
    playBtn.dataset.mode = 'pause';
  }
});

$('prev').addEventListener('click', () => step(-1));
$('next').addEventListener('click', () => step(1));

function step(dir) {
  if (!payload) return;
  const cur = sim.segments.findIndex((s) => sim.time < s.end);
  sim.seekSegment((cur < 0 ? 0 : cur) + dir);
  sim.pause();
  playBtn.textContent = 'Play';
  playBtn.dataset.mode = 'play';
}

scrub.addEventListener('input', () => {
  if (!payload) return;
  sim.pause();
  playBtn.textContent = 'Play';
  playBtn.dataset.mode = 'play';
  sim.setTime((Number(scrub.value) / 1000) * sim.duration);
});

speedSel.addEventListener('change', () => { sim.speed = Number(speedSel.value); });

glowSel.addEventListener('change', () => { sim.setGlow(Number(glowSel.value)); });

recenterBtn.addEventListener('click', () => {
  sim.recenter();
  recenterBtn.classList.remove('is-on');
});

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); playBtn.click(); }
  if (e.code === 'ArrowRight') step(1);
  if (e.code === 'ArrowLeft') step(-1);
  if (e.key === 'r' || e.key === 'R') { sim.replay(); playBtn.textContent = 'Pause'; playBtn.dataset.mode = 'pause'; }
});

/* ---------- hover tooltips ---------- */

const tooltip = $('tooltip');
const tipTitle = $('tip-title');
const tipBody = $('tip-body');
let tipRaf = 0;
let dragging = false;

function hideTip() {
  tooltip.classList.remove('is-on');
  tooltip.setAttribute('aria-hidden', 'true');
  if (sim) sim.clearHover();
}

if (sim) {
  const canvas = sim.canvas;

  canvas.addEventListener('pointerdown', () => { dragging = true; hideTip(); });
  window.addEventListener('pointerup', () => { dragging = false; });
  canvas.addEventListener('pointerleave', hideTip);

  canvas.addEventListener('pointermove', (e) => {
    if (dragging || !payload) return;
    // At most one raycast per animation frame, so a fast pointer cannot
    // outrun the renderer.
    if (tipRaf) return;
    tipRaf = requestAnimationFrame(() => {
      tipRaf = 0;
      const r = canvas.getBoundingClientRect();
      const tip = sim.pickAt(
        ((e.clientX - r.left) / r.width) * 2 - 1,
        -((e.clientY - r.top) / r.height) * 2 + 1,
      );
      if (!tip) { hideTip(); return; }
      tipTitle.textContent = tip.title;
      tipBody.innerHTML = '';
      tip.lines.forEach((line) => {
        const p = document.createElement('p');
        p.textContent = line;
        tipBody.appendChild(p);
      });
      tooltip.classList.add('is-on');
      tooltip.setAttribute('aria-hidden', 'false');
      // Flip near the edges so the card never leaves the window.
      const w = tooltip.offsetWidth || 260;
      const hgt = tooltip.offsetHeight || 90;
      const x = e.clientX + 18 + w > window.innerWidth ? e.clientX - w - 18 : e.clientX + 18;
      const y = e.clientY + 16 + hgt > window.innerHeight ? e.clientY - hgt - 16 : e.clientY + 16;
      tooltip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
    });
  });

  // The camera hint fades once the user has actually moved the camera.
  const hint = $('hint');
  const fadeHint = () => hint.classList.add('is-gone');
  canvas.addEventListener('pointerdown', fadeHint, { once: true });
  canvas.addEventListener('wheel', fadeHint, { once: true });
}

/* ---------- record the simulation to a video file ---------- */

recordBtn.addEventListener('click', () => {
  if (!payload) return;
  if (recorder && recorder.state === 'recording') {
    recorder.stop();
    return;
  }
  const stream = sim.canvas.captureStream(60);
  const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  const mime = types.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
  if (!mime) {
    setStatus('This browser cannot record. Use Chrome or Edge, or screen-record instead.', 'error');
    return;
  }
  chunks = [];
  recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12_000_000 });
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: 'video/webm' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'netsim3d-simulation.webm';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    recordBtn.textContent = 'Record video';
    recordBtn.classList.remove('is-rec');
    setStatus('Saved netsim3d-simulation.webm to your downloads.', 'info');
    setTimeout(() => setStatus(''), 4000);
  };
  recorder.start();
  recordBtn.textContent = 'Stop and save';
  recordBtn.classList.add('is-rec');
  sim.replay();
  playBtn.textContent = 'Pause';
  playBtn.dataset.mode = 'pause';
  const stopAt = (sim.duration / sim.speed + 0.8) * 1000;
  setTimeout(() => { if (recorder && recorder.state === 'recording') recorder.stop(); }, stopAt);
});
