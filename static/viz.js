/* NetSim3D — 3D scene + timeline.
 *
 * Consumes the JSON payload from /api/analyze and turns it into a stack of
 * glowing layers you fly through. Everything is driven by one clock, so the
 * same code powers playback, scrubbing, stage-stepping and video recording.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const SPACING = 14;       // world units between layers
const TILE = 2.0;         // one feature map tile
const GAP = 0.3;
const INPUT_W = 9;        // width of the input image plane

// One wheel notch moves this fraction of the whole zoom range, about 4% of
// the current distance. Deliberately gentler than the OrbitControls default,
// and because the range is stepped logarithmically a notch feels identical
// whether you are inches from a tile or looking at the entire network.
// Large moves are what the chapter chips and Re-centre are for.
const ZOOM_STEP = 0.0085;

/* ---------- colour ramps ---------- */

function lerp3(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

// deep indigo -> cyan -> white hot
const RAMP = [[8, 12, 40], [26, 92, 176], [34, 178, 206], [176, 232, 244]];

function activationColor(t) {
  const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x));
  return lerp3(RAMP[i], RAMP[i + 1], x - i);
}

const HEAT = [[0, 0, 0], [96, 24, 0], [222, 148, 26], [246, 226, 188]];

function heatColor(t) {
  const x = Math.max(0, Math.min(1, t)) * (HEAT.length - 1);
  const i = Math.min(HEAT.length - 2, Math.floor(x));
  return lerp3(HEAT[i], HEAT[i + 1], x - i);
}

/* ---------- small helpers ---------- */

const smooth = (t) => t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
const clamp01 = (t) => Math.max(0, Math.min(1, t));

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function mapTexture(bytes, offset, res, ramp) {
  const rgba = new Uint8Array(res * res * 4);
  for (let y = 0; y < res; y++) {
    const srcRow = (res - 1 - y) * res;       // flip vertically
    for (let x = 0; x < res; x++) {
      const v = bytes[offset + srcRow + x] / 255;
      const c = ramp(v);
      const p = (y * res + x) * 4;
      rgba[p] = c[0]; rgba[p + 1] = c[1]; rgba[p + 2] = c[2];
      rgba[p + 3] = Math.round(255 * Math.pow(v, 0.9));
    }
  }
  const tex = new THREE.DataTexture(rgba, res, res, THREE.RGBAFormat);
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

function labelSprite(lines, opts = {}) {
  const pad = 24;
  const scale = 2;
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');
  const sizes = lines.map((l, i) => (i === 0 ? 46 : 32));
  ctx.font = `600 ${sizes[0]}px "IBM Plex Sans", system-ui, sans-serif`;
  let w = 0;
  lines.forEach((l, i) => {
    ctx.font = `${i === 0 ? '600' : '400'} ${sizes[i]}px ${i === 0 ? '"IBM Plex Sans", system-ui, sans-serif' : '"IBM Plex Mono", ui-monospace, monospace'}`;
    w = Math.max(w, ctx.measureText(l).width);
  });
  const lineH = 1.45;
  c.width = Math.ceil((w + pad * 2) * scale);
  c.height = Math.ceil((sizes.reduce((a, b) => a + b * lineH, 0) + pad * 2) * scale);
  ctx.scale(scale, scale);
  ctx.textBaseline = 'top';
  let y = pad;
  lines.forEach((l, i) => {
    ctx.font = `${i === 0 ? '600' : '400'} ${sizes[i]}px ${i === 0 ? '"IBM Plex Sans", system-ui, sans-serif' : '"IBM Plex Mono", ui-monospace, monospace'}`;
    ctx.fillStyle = i === 0 ? (opts.color || '#dff6ff') : '#6fb6cc';
    ctx.fillText(l, pad, y);
    y += sizes[i] * lineH;
  });
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({
    map: tex, transparent: true, depthWrite: false, depthTest: false,
    toneMapped: false, fog: false,
  });
  const sp = new THREE.Sprite(mat);
  const world = opts.width || 5.2;
  sp.scale.set(world, world * (c.height / c.width), 1);
  sp.userData.fade = mat;
  return sp;
}

function corners(size, z, color) {
  const h = size / 2;
  const arm = size * 0.13;
  const pts = [];
  const push = (ax, ay) => {
    pts.push(ax * h, ay * h, z, ax * h - ax * arm, ay * h, z);
    pts.push(ax * h, ay * h, z, ax * h, ay * h - ay * arm, z);
  };
  push(1, 1); push(-1, 1); push(1, -1); push(-1, -1);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  const m = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 });
  return new THREE.LineSegments(g, m);
}

/* ---------- the scene ---------- */

export class Sim {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#04060c');
    // Fog range is recomputed in build() from the real depth of the stack.
    // A fixed range is a trap: normal-blended surfaces past `far` turn into
    // solid background-coloured rectangles, so the scene goes black the
    // moment the user scrolls out past it.
    this.scene.fog = new THREE.Fog('#04060c', 400, 1200);

    this.camera = new THREE.PerspectiveCamera(46, 1, 0.1, 4000);
    this.camera.position.set(10, 5, 22);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.9;
    container.appendChild(this.renderer.domElement);
    this.canvas = this.renderer.domElement;

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 2.5;
    this.controls.maxDistance = 900;      // tightened in build()
    // OrbitControls dollies by a fixed percentage, instantly, once per wheel
    // event. That reads as a jump when far out and as nothing when close in,
    // and high-resolution wheels fire several events per notch. Drive it here
    // instead.
    this.controls.enableZoom = false;
    this.zoomTarget = null;
    this.manual = false;
    this.controls.addEventListener('start', () => {
      this.manual = true;
      if (this.onManual) this.onManual();
    });

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.30, 0.45, 0.72);
    this.glow = 0.4;
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.scene.add(new THREE.AmbientLight(0x88aaff, 0.6));
    const key = new THREE.PointLight(0x66ccff, 40, 120);
    key.position.set(8, 10, 12);
    this.scene.add(key);

    this.root = new THREE.Group();
    this.scene.add(this.root);

    this._ray = new THREE.Raycaster();
    this._ndc = new THREE.Vector2();

    this.canvas.addEventListener('wheel', (e) => this._onWheel(e), { passive: false });
    this._pinch = null;
    this.canvas.addEventListener('touchstart', (e) => this._onTouch(e), { passive: true });
    this.canvas.addEventListener('touchmove', (e) => this._onTouch(e), { passive: true });
    this.canvas.addEventListener('touchend', () => { this._pinch = null; }, { passive: true });

    // Outline that marks whatever the pointer is over. Kept outside `root` so
    // rebuilding the scene never disposes it.
    const ringPts = [];
    const r = 1;
    const corners2 = [[-r, -r], [r, -r], [r, r], [-r, r]];
    for (let i = 0; i < 4; i++) {
      const a = corners2[i], b = corners2[(i + 1) % 4];
      ringPts.push(a[0], a[1], 0, b[0], b[1], 0);
    }
    const ringGeo = new THREE.BufferGeometry();
    ringGeo.setAttribute('position', new THREE.Float32BufferAttribute(ringPts, 3));
    this.hoverRing = new THREE.LineSegments(ringGeo, new THREE.LineBasicMaterial({
      color: 0xffd479, transparent: true, opacity: 0.95, depthTest: false, fog: false,
    }));
    this.hoverRing.visible = false;
    this.hoverRing.renderOrder = 999;
    this.scene.add(this.hoverRing);

    this.layers = [];
    this.segments = [];
    this.duration = 1;
    this.time = 0;
    this.playing = false;
    this.speed = 1;
    this.clock = new THREE.Clock();
    this.pulses = [];
    this.payload = null;
    this.follow = true;          // manual camera still tracks the active stage
    this.pickables = [];         // objects the hover tooltip can hit

    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);
    this.resize();
    this._loop();
  }

  resize() {
    const w = this.container.clientWidth || 800;
    const h = this.container.clientHeight || 600;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.bloom.setSize(w, h);
  }

  clear() {
    this.root.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((m) => {
          if (m.map && m.map.dispose) m.map.dispose();
          if (m.dispose) m.dispose();
        });
      }
    });
    this.root.clear();
    this.layers = [];
    this.pulses = [];
    this.segments = [];
    this.time = 0;
    this.input = null;
    this.vector = null;
    this.output = null;
    this.camMat = null;
    this.kernel = null;
    this.beam = null;
    this.payload = null;
    this.pickables = [];
    this.hovered = null;
    if (this.hoverRing) this.hoverRing.visible = false;
  }

  /* ---- build ---- */

  build(payload) {
    this.clear();
    this.payload = payload;

    const convLayers = payload.layers.filter((l) => l.kind === 'features');
    const vectorLayer = payload.layers.find((l) => l.kind === 'vector');
    const outLayer = payload.layers.find((l) => l.kind === 'output');

    /* input plane */
    const tex = new THREE.TextureLoader().load(payload.input.png);
    tex.colorSpace = THREE.SRGBColorSpace;
    const inputGroup = new THREE.Group();
    const inputMat = new THREE.MeshBasicMaterial({
      map: tex, transparent: true, opacity: 0, side: THREE.DoubleSide,
    });
    const inputMesh = new THREE.Mesh(new THREE.PlaneGeometry(INPUT_W, INPUT_W), inputMat);
    this._pick(inputMesh, INPUT_W, 'Your image', [
      `${payload.input.size} x ${payload.input.size} x 3, resized and centre cropped`,
      'Normalised with the training mean and standard deviation. This tensor, '
      + 'not the original file, is what the first layer reads.',
    ]);
    inputGroup.add(inputMesh, corners(INPUT_W * 1.06, 0.01, 0x2fd2ee));
    const inLabel = labelSprite(['your image', `${payload.input.size}x${payload.input.size}x3`], { width: 4.6 });
    inLabel.position.set(0, INPUT_W / 2 + 1.5, 0);
    inLabel.userData.fade.opacity = 0;
    inputGroup.add(inLabel);
    this.root.add(inputGroup);
    this.input = { group: inputGroup, mat: inputMat, label: inLabel, z: 0 };

    /* kernel window that slides over the image during stage 1 */
    const kg = new THREE.BufferGeometry();
    const k = 0.62;
    kg.setAttribute('position', new THREE.Float32BufferAttribute([
      -k, -k, 0, k, -k, 0, k, -k, 0, k, k, 0, k, k, 0, -k, k, 0, -k, k, 0, -k, -k, 0,
    ], 3));
    this.kernel = new THREE.LineSegments(kg, new THREE.LineBasicMaterial({
      color: 0xfff0b0, transparent: true, opacity: 0,
    }));
    this.root.add(this.kernel);
    const beamGeo = new THREE.BufferGeometry();
    beamGeo.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, -1], 3));
    this.beam = new THREE.Line(beamGeo, new THREE.LineBasicMaterial({
      color: 0xffd479, transparent: true, opacity: 0,
    }));
    this.root.add(this.beam);

    /* grad-cam overlay, sits just in front of the input */
    const attr = payload.attribution;
    if (attr && attr.kind === 'heatmap') {
      const bytes = b64ToBytes(attr.data);
      const camTex = mapTexture(bytes, 0, attr.res, heatColor);
      this.camMat = new THREE.MeshBasicMaterial({
        map: camTex, transparent: true, opacity: 0, depthWrite: false,
        side: THREE.DoubleSide,
      });
      const camMesh = new THREE.Mesh(new THREE.PlaneGeometry(INPUT_W, INPUT_W), this.camMat);
      camMesh.position.z = 0.08;
      this._pick(camMesh, INPUT_W, `Attribution (${attr.method})`, [
        'Bright regions are the pixels that pushed the winning class up.',
        'If the heat sits away from the object, the answer was luck rather '
        + 'than evidence.',
      ]);
      this.root.add(camMesh);
    }

    /* conv stages */
    convLayers.forEach((layer, li) => {
      const z = -SPACING * (li + 1);
      const group = new THREE.Group();
      group.position.z = z;
      const bytes = b64ToBytes(layer.data);
      const cols = Math.ceil(Math.sqrt(layer.shown));
      const rows = Math.ceil(layer.shown / cols);
      const width = cols * (TILE + GAP) - GAP;
      const height = rows * (TILE + GAP) - GAP;
      const tiles = [];

      for (let i = 0; i < layer.shown; i++) {
        const cx = (i % cols) * (TILE + GAP) - width / 2 + TILE / 2;
        const cy = -Math.floor(i / cols) * (TILE + GAP) + height / 2 - TILE / 2;
        const t = mapTexture(bytes, i * layer.res * layer.res, layer.res, activationColor);
        const mat = new THREE.MeshBasicMaterial({
          map: t, transparent: true, opacity: 0, depthWrite: false,
          side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(TILE, TILE), mat);
        mesh.position.set(cx, cy, 0);
        mesh.scale.setScalar(0.02);
        group.add(mesh);
        this._pick(mesh, TILE, `${layer.title} · filter ${layer.channels[i]}`, [
          `filter ${layer.channels[i]} of ${layer.shape[0]}, shown at `
          + `${layer.res}x${layer.res} (true size ${layer.shape[1]}x${layer.shape[2]})`,
          'Bright pixels are where this filter found the pattern it learned. '
          + 'Dark means it stayed quiet there.',
          `Rank ${i + 1} by average activation on your input.`,
        ]);
        tiles.push({ mesh, mat, delay: i / layer.shown });
      }

      group.add(corners(Math.max(width, height) * 1.1, -0.02, 0x1b5f80));

      const label = labelSprite([layer.title,
        `${layer.shape[0]}x${layer.shape[1]}x${layer.shape[2]}`], { width: 5.4 });
      label.position.set(0, height / 2 + 1.7 + (li % 2 ? 1.6 : 0), 0);
      label.userData.fade.opacity = 0;
      group.add(label);

      /* threads from the previous stage */
      const prevZ = li === 0 ? 0 : -SPACING * li;
      const prevHalf = li === 0 ? INPUT_W / 2 : width / 2;
      const pos = [];
      for (let i = 0; i < layer.shown; i++) {
        const t = tiles[i].mesh.position;
        for (let j = 0; j < 2; j++) {
          pos.push((Math.random() * 2 - 1) * prevHalf, (Math.random() * 2 - 1) * prevHalf, prevZ - z);
          pos.push(t.x, t.y, 0);
        }
      }
      const lg = new THREE.BufferGeometry();
      lg.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      const lmat = new THREE.LineBasicMaterial({
        color: 0x1d6c8e, transparent: true, opacity: 0, depthWrite: false,
      });
      const threads = new THREE.LineSegments(lg, lmat);
      group.add(threads);

      this.root.add(group);
      this.layers.push({ spec: layer, group, tiles, label, threads: lmat, z, width, height });
    });

    /* pooled fingerprint */
    let z = -SPACING * (convLayers.length + 1);
    if (vectorLayer) {
      const group = new THREE.Group();
      group.position.z = z;
      const n = vectorLayer.shown;
      const bw = 0.09, bgap = 0.035;
      const total = n * (bw + bgap);
      const bars = [];
      for (let i = 0; i < n; i++) {
        const v = vectorLayer.values[i];
        const hgt = Math.max(0.06, Math.abs(v) * 4.2);
        const c = v >= 0 ? activationColor(0.35 + Math.abs(v) * 0.65) : [190, 70, 140];
        const mat = new THREE.MeshBasicMaterial({
          color: new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255),
          transparent: true, opacity: 0, depthWrite: false,
          side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(bw, hgt), mat);
        mesh.position.set(i * (bw + bgap) - total / 2, hgt / 2 * (v >= 0 ? 1 : -1), 0);
        mesh.scale.y = 0.02;
        group.add(mesh);
        this._pick(mesh, 0.6, `Pooled unit ${vectorLayer.index[i]}`, [
          `value ${v.toFixed(3)} (normalised), unit ${vectorLayer.index[i]} of `
          + `${vectorLayer.shape[0]}`,
          'One whole feature map averaged down to this single number. '
          + 'Position in the image is gone by this point.',
        ]);
        bars.push({ mesh, mat, delay: i / n });
      }
      const label = labelSprite([vectorLayer.title, `${vectorLayer.shape[0]} numbers`], { width: 5.6 });
      label.position.set(0, 4.4, 0);
      label.userData.fade.opacity = 0;
      group.add(label);
      this.root.add(group);
      this.vector = { spec: vectorLayer, group, bars, label, z };
      z -= SPACING;
    }

    /* classifier bars */
    if (outLayer) {
      const group = new THREE.Group();
      group.position.z = z;
      const top = outLayer.top;
      const rowH = 1.25;
      const maxW = 9.5;
      const bars = [];
      top.forEach((entry, i) => {
        const win = i === 0;
        const col = win ? new THREE.Color(1.0, 0.72, 0.22) : new THREE.Color(0.14, 0.72, 0.86);
        const mat = new THREE.MeshBasicMaterial({
          color: col, transparent: true, opacity: 0, depthWrite: false,
          side: THREE.DoubleSide,
        });
        const w = Math.max(0.12, entry.p) * maxW;
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, 0.5), mat);
        const y = (top.length / 2 - i) * rowH;
        mesh.position.set(-maxW / 2 + w / 2, y, 0);
        mesh.scale.x = 0.001;
        group.add(mesh);
        this._pick(mesh, Math.max(w, 1.2), entry.label, [
          `${(entry.p * 100).toFixed(2)}% after softmax`,
          `rank ${i + 1} of ${outLayer.shape[0]} classes`,
          win ? 'This is the model\u2019s answer.'
              : 'Considered, but scored lower than the winner.',
        ]);
        const lab = labelSprite([`${entry.label}`, `${(entry.p * 100).toFixed(1)}%`], {
          width: 4.4, color: win ? '#ffd479' : '#cfeaf5',
        });
        lab.position.set(maxW / 2 + 2.6, y, 0);
        lab.userData.fade.opacity = 0;
        group.add(lab);
        bars.push({ mesh, mat, lab, delay: i * 0.12, w });
      });
      const label = labelSprite([outLayer.title, `${outLayer.shape[0]} classes -> softmax`], { width: 6.0 });
      label.position.set(0, (top.length / 2 + 1.3) * rowH, 0);
      label.userData.fade.opacity = 0;
      group.add(label);
      this.root.add(group);
      this.output = { spec: outLayer, group, bars, label, z };
    }

    this._fitToScene();
    this._buildTimeline();
    this.setTime(0);
    return this.segments;
  }

  /** Size fog, clipping and zoom limits from the scene that actually exists.
   *  Fixed values are a trap: they break the moment a model has more stages,
   *  and a normal-blended surface past fog `far` becomes a solid
   *  background-coloured rectangle, which reads to the user as "it went black". */
  _fitToScene() {
    const deepest = this.output ? this.output.z
      : this.vector ? this.vector.z
        : this.layers.length ? this.layers[this.layers.length - 1].z : -SPACING;
    const depth = Math.abs(deepest) + INPUT_W * 3;
    this.sceneDepth = depth;
    this.deepestZ = deepest;

    // 1. How far out may the user go?
    const maxOut = depth * 1.9;
    this.controls.minDistance = 2.0;
    this.controls.maxDistance = maxOut;

    // 2. Fog may only begin beyond that, or the scene fades to background
    //    colour exactly when someone zooms out to see all of it.
    this.scene.fog.near = maxOut * 1.2;   // margin, so fog never bites at full zoom-out
    this.scene.fog.far = maxOut * 4.0;

    // 3. The far plane clears the fog, so nothing is ever clipped away.
    this.camera.near = 0.1;
    this.camera.far = this.scene.fog.far * 2;
    this.camera.updateProjectionMatrix();
  }

  /** Panning moves the pivot, so zoom limits alone cannot stop someone
   *  sliding the whole scene off screen. Keep the pivot inside a box around
   *  the network and carry the camera with it, so the view is always
   *  recoverable no matter what the user does with the mouse. */
  _clampView() {
    if (!this.sceneDepth) return;
    const lim = this.sceneDepth * 0.9;
    const cz = this.deepestZ / 2;
    const t = this.controls.target;
    const nx = Math.min(lim, Math.max(-lim, t.x));
    const ny = Math.min(lim * 0.6, Math.max(-lim * 0.6, t.y));
    const nz = Math.min(cz + lim, Math.max(cz - lim, t.z));
    if (nx !== t.x || ny !== t.y || nz !== t.z) {
      const dx = nx - t.x, dy = ny - t.y, dz = nz - t.z;
      t.set(nx, ny, nz);
      this.camera.position.set(this.camera.position.x + dx,
        this.camera.position.y + dy, this.camera.position.z + dz);
    }
  }

  /* ---- zoom ---- */

  _camDistance() {
    const t = this.controls.target;
    const p = this.camera.position;
    const dx = p.x - t.x, dy = p.y - t.y, dz = p.z - t.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
  }

  /** Move the zoom target by whole notches. Positive ticks pull back.
   *  Distance is stepped in log space, so every notch covers the same share
   *  of the range no matter where the camera currently is. */
  applyZoomTicks(ticks) {
    if (!Number.isFinite(ticks) || ticks === 0) return;
    const lo = Math.log(this.controls.minDistance);
    const hi = Math.log(this.controls.maxDistance);
    const cur = this.zoomTarget == null ? this._camDistance() : this.zoomTarget;
    const u = (Math.log(cur) - lo) / (hi - lo);
    const next = Math.max(0, Math.min(1, u + ticks * ZOOM_STEP));
    this.zoomTarget = Math.exp(lo + next * (hi - lo));
    if (!this.manual) {
      this.manual = true;
      if (this.onManual) this.onManual();
    }
  }

  _onWheel(e) {
    e.preventDefault();
    // deltaMode 0 is pixels, 1 is lines, 2 is pages. One notch of a normal
    // mouse is 100 pixels. Clamping stops a fast flick or a trackpad burst
    // from throwing the camera across the scene in one frame.
    const raw = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 100 : 1);
    this.applyZoomTicks(Math.max(-2.5, Math.min(2.5, raw / 100)));
  }

  _onTouch(e) {
    if (!e.touches || e.touches.length !== 2) { this._pinch = null; return; }
    const [a, b] = e.touches;
    const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    if (this._pinch == null) { this._pinch = d; return; }
    if (d > 1 && this._pinch > 1) {
      // A pinch that halves the finger gap equals about ten notches out.
      this.applyZoomTicks(Math.log(this._pinch / d) * 24);
    }
    this._pinch = d;
  }

  /** Ease the camera towards the zoom target, frame-rate independent. */
  _stepZoom(dt) {
    if (this.zoomTarget == null) return;
    const cur = this._camDistance();
    const k = 1 - Math.pow(0.000004, Math.max(0.001, dt));   // ~0.4s to settle
    const next = cur + (this.zoomTarget - cur) * k;
    const t = this.controls.target;
    const p = this.camera.position;
    const s = next / cur;
    this.camera.position.set(t.x + (p.x - t.x) * s,
      t.y + (p.y - t.y) * s, t.z + (p.z - t.z) * s);
    if (Math.abs(this.zoomTarget - next) < this.zoomTarget * 5e-4) this.zoomTarget = null;
  }

  /* ---- hover picking ---- */

  /** Tag an object as hoverable. size is its world width, used by the ring. */
  _pick(mesh, size, title, lines) {
    mesh.userData.tip = { title, lines };
    mesh.userData.pickSize = size;
    this.pickables.push(mesh);
  }

  /** Raycast at normalised device coordinates. Returns a tip or null.
   *  Objects that have not faded in yet are skipped, so the user never gets a
   *  tooltip for something they cannot see. */
  pickAt(ndcX, ndcY) {
    if (!this.pickables.length) return null;
    this._ray.setFromCamera(this._ndc.set(ndcX, ndcY), this.camera);
    const visible = this.pickables.filter(
      (m) => m.visible && m.material && m.material.opacity > 0.06);
    const hits = this._ray.intersectObjects(visible, false);
    const hit = hits && hits.length ? hits[0] : null;
    this._setHover(hit ? hit.object : null);
    return hit ? hit.object.userData.tip : null;
  }

  clearHover() { this._setHover(null); }

  _setHover(obj) {
    if (this.hovered === obj) return;
    this.hovered = obj;
    if (!this.hoverRing) return;
    if (!obj) {
      this.hoverRing.visible = false;
      return;
    }
    const size = obj.userData.pickSize || 1;
    obj.getWorldPosition(this.hoverRing.position);
    this.hoverRing.quaternion.copy(this.camera.quaternion);
    this.hoverRing.scale.setScalar(size * 0.62);
    this.hoverRing.visible = true;
  }

  /* ---- timeline ---- */

  _buildTimeline() {
    const segs = [];
    const add = (id, title, dur, layerId) => {
      const start = segs.length ? segs[segs.length - 1].end : 0;
      segs.push({ id, title, start, end: start + dur, dur, layerId: layerId || null });
    };
    add('input', 'Image in', 2.6);
    this.layers.forEach((l, i) =>
      add(`conv${i}`, l.spec.title, i === 0 ? 4.0 : 3.0, l.spec.id));
    if (this.vector) add('vector', this.vector.spec.title, 2.8, this.vector.spec.id);
    if (this.output) add('output', 'Decision', 3.6, this.output.spec.id);
    if (this.camMat) add('cam', 'Where it looked', 3.2);
    add('outro', 'Whole network', 4.4);
    this.segments = segs;
    this.duration = segs[segs.length - 1].end;
  }

  segmentAt(t) {
    for (const s of this.segments) if (t < s.end) return s;
    return this.segments[this.segments.length - 1];
  }

  seekSegment(i) {
    const s = this.segments[Math.max(0, Math.min(this.segments.length - 1, i))];
    this.setTime(s.start + 0.001);
    return s;
  }

  setTime(t) {
    this.time = Math.max(0, Math.min(this.duration, t));
    this._apply(this.time);
  }

  _camFor(t) {
    const seg = this.segmentAt(t);
    const k = clamp01((t - seg.start) / seg.dur);
    const e = smooth(k);
    const pos = new THREE.Vector3();
    const tgt = new THREE.Vector3();

    if (seg.id === 'input') {
      pos.set(14 - 6 * e, 7 - 3.5 * e, 30 - 11 * e);
      tgt.set(0, 0, -2 * e);
    } else if (seg.id.startsWith('conv')) {
      const i = parseInt(seg.id.slice(4), 10);
      const z = this.layers[i].z;
      const side = i % 2 === 0 ? 1 : -1;
      pos.set(side * (7.5 - 4.5 * e), 3.4 - 1.6 * e, z + 13.5 - 9.5 * e);
      tgt.set(0, 0, z - 1.5 * e);
    } else if (seg.id === 'vector') {
      const z = this.vector.z;
      pos.set(-8 + 5 * e, 2.4, z + 11 - 5 * e);
      tgt.set(0, 0.5, z);
    } else if (seg.id === 'output') {
      const z = this.output.z;
      pos.set(2 + 2 * e, 0.6, z + 16 - 5 * e);
      tgt.set(1.2, 0.4, z);
    } else if (seg.id === 'cam') {
      const zEnd = 0;
      const from = this.output ? this.output.z : this.layers[this.layers.length - 1].z;
      pos.set(0.5, 1.0, from + 16 + (zEnd + 15 - (from + 16)) * e);
      tgt.set(0, 0, from * (1 - e));
    } else {
      const deep = this.output ? this.output.z : -SPACING * this.layers.length;
      pos.set(26 + 10 * e, 13 + 4 * e, 26 + deep * 0.45 * e);
      tgt.set(0, 0, deep * 0.5);
    }
    return { pos, tgt };
  }

  _apply(t) {
    if (!this.input || !this.segments.length) return;
    const seg = this.segmentAt(t);
    const k = clamp01((t - seg.start) / seg.dur);

    /* input */
    const inAlive = t > 0.15 ? smooth(clamp01((t - 0.15) / 1.2)) : 0;
    const inputStage = seg.id === 'input' || seg.id === 'cam' || seg.id === 'outro';
    this.input.mat.opacity = (0.12 + 0.88 * inAlive) * (inputStage ? 1 : 0.7);
    this.input.label.userData.fade.opacity = inAlive * (inputStage ? 1 : 0.25);

    /* stages */
    this.layers.forEach((l, i) => {
      const sg = this.segments.find((s) => s.id === `conv${i}`);
      const local = (t - sg.start) / sg.dur;
      const on = clamp01(local * 1.9);
      // Past stages stay visible but step back, so the stage being narrated
      // is the brightest thing on screen and the wide shot reads as depth
      // rather than one lump of light.
      const past = t > sg.end;
      l.threads.opacity = 0.11 * smooth(clamp01(local * 3)) * (past ? 0.35 : 1);
      l.label.userData.fade.opacity =
        smooth(clamp01((local - 0.05) * 3)) * (past ? 0.28 : 1);
      l.tiles.forEach((tile) => {
        const a = smooth(clamp01((on - tile.delay * 0.55) * 2.2));
        tile.mat.opacity = a * (past ? 0.5 : 0.95);
        tile.mesh.scale.setScalar(0.02 + 0.98 * a);
      });
    });

    /* kernel sweep during the first stage */
    if (this.layers.length && this.kernel) {
      const sg = this.segments[1];
      const inStage = t >= sg.start && t <= sg.end;
      const local = clamp01((t - sg.start) / sg.dur);
      const vis = inStage ? Math.sin(Math.min(1, local / 0.75) * Math.PI) : 0;
      this.kernel.material.opacity = vis;
      this.beam.material.opacity = vis * 0.8;
      if (vis > 0.01) {
        const steps = 7;
        const p = Math.min(0.999, local / 0.75) * steps * steps;
        const row = Math.floor(p / steps);
        const colRaw = p % steps;
        const col = row % 2 === 0 ? colRaw : steps - 1 - colRaw;
        const half = INPUT_W / 2 - 0.7;
        const kx = -half + (col / (steps - 1)) * half * 2;
        const ky = half - (row / (steps - 1)) * half * 2;
        this.kernel.position.set(kx, ky, 0.06);
        const target = this.layers[0].tiles[0].mesh;
        const pa = this.beam.geometry.attributes.position;
        pa.setXYZ(0, kx, ky, 0.06);
        pa.setXYZ(1, target.position.x, target.position.y, this.layers[0].z);
        pa.needsUpdate = true;
      }
    }

    /* fingerprint */
    if (this.vector) {
      const sg = this.segments.find((s) => s.id === 'vector');
      const local = (t - sg.start) / sg.dur;
      this.vector.label.userData.fade.opacity = smooth(clamp01(local * 3));
      this.vector.bars.forEach((b) => {
        const a = smooth(clamp01((local * 1.6 - b.delay * 0.8) * 2));
        b.mat.opacity = a * (t > sg.end ? 0.45 : 0.95);
        b.mesh.scale.y = 0.02 + 0.98 * a;
      });
    }

    /* decision */
    if (this.output) {
      const sg = this.segments.find((s) => s.id === 'output');
      const local = (t - sg.start) / sg.dur;
      this.output.label.userData.fade.opacity = smooth(clamp01(local * 3));
      this.output.bars.forEach((b) => {
        const a = smooth(clamp01((local * 1.5 - b.delay) * 1.8));
        b.mat.opacity = 0.92 * a;
        b.mesh.scale.x = 0.001 + 0.999 * a;
        b.mesh.position.x = -9.5 / 2 + (b.w * (0.001 + 0.999 * a)) / 2;
        b.lab.userData.fade.opacity = a;
      });
    }

    /* grad-cam */
    if (this.camMat) {
      const sg = this.segments.find((s) => s.id === 'cam');
      const local = clamp01((t - sg.start) / sg.dur);
      this.camMat.opacity = t < sg.start ? 0 : smooth(Math.min(1, local * 2)) * 0.6;
    }

    /* camera */
    const { pos, tgt } = this._camFor(t);
    if (!this.manual) {
      this.camera.position.copy(pos);
      this.controls.target.copy(tgt);
    } else if (this.follow && this.playing) {
      // The user has taken the camera. Keep their angle and distance, but
      // drift the pivot towards the stage being narrated, so they are never
      // left staring at empty space while the run continues behind them.
      const dx = (tgt.x - this.controls.target.x) * 0.05;
      const dy = (tgt.y - this.controls.target.y) * 0.05;
      const dz = (tgt.z - this.controls.target.z) * 0.05;
      this.controls.target.set(this.controls.target.x + dx,
        this.controls.target.y + dy, this.controls.target.z + dz);
      this.camera.position.set(this.camera.position.x + dx,
        this.camera.position.y + dy, this.camera.position.z + dz);
    }

    if (this.onSegment) this.onSegment(seg, k, t);
  }

  /* ---- pulses along the threads ---- */

  _spawnPulses(seg) {
    if (!seg.id.startsWith('conv')) return;
    const i = parseInt(seg.id.slice(4), 10);
    const l = this.layers[i];
    if (!l || l.pulseDone) return;
    l.pulseDone = true;
    const fromZ = i === 0 ? 0 : this.layers[i - 1].z;
    const geo = new THREE.SphereGeometry(0.075, 6, 6);
    const mat = new THREE.MeshBasicMaterial({
      color: 0x8fe4f7, transparent: true, opacity: 0.5,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    for (let n = 0; n < 40; n++) {
      const tile = l.tiles[n % l.tiles.length].mesh;
      const half = (i === 0 ? INPUT_W : l.width) / 2;
      const m = new THREE.Mesh(geo, mat);
      this.root.add(m);
      this.pulses.push({
        mesh: m,
        from: new THREE.Vector3((Math.random() * 2 - 1) * half, (Math.random() * 2 - 1) * half, fromZ),
        to: new THREE.Vector3(tile.position.x, tile.position.y, l.z),
        t: -Math.random(),
        speed: 0.55 + Math.random() * 0.5,
      });
    }
  }

  _stepPulses(dt) {
    for (let i = this.pulses.length - 1; i >= 0; i--) {
      const p = this.pulses[i];
      p.t += dt * p.speed;
      if (p.t >= 1) {
        this.root.remove(p.mesh);
        this.pulses.splice(i, 1);
        continue;
      }
      const a = Math.max(0, p.t);
      p.mesh.position.lerpVectors(p.from, p.to, a);
      p.mesh.visible = p.t > 0;
      p.mesh.scale.setScalar(0.6 + Math.sin(a * Math.PI) * 0.9);
    }
  }

  /* ---- loop ---- */

  /** 0 = flat and readable, 1 = heavy neon. 0.4 is the default. */
  setGlow(v) {
    this.glow = Math.max(0, Math.min(1, v));
    this.bloom.strength = 0.10 + this.glow * 0.62;
    this.bloom.radius = 0.32 + this.glow * 0.45;
    this.bloom.threshold = 0.86 - this.glow * 0.32;
    this.renderer.toneMappingExposure = 0.80 + this.glow * 0.30;
  }

  play() { this.playing = true; }
  pause() { this.playing = false; }

  replay() {
    this.layers.forEach((l) => { l.pulseDone = false; });
    this.manual = false;
    this.zoomTarget = null;
    this.setTime(0);
    this.playing = true;
  }

  recenter() {
    this.manual = false;
    this.zoomTarget = null;
    this._apply(this.time);
  }

  _loop() {
    const dt = Math.min(0.05, this.clock.getDelta());
    if (this.playing && this.payload) {
      const seg = this.segmentAt(this.time);
      this._spawnPulses(seg);
      const next = this.time + dt * this.speed;
      if (next >= this.duration) {
        this.setTime(this.duration);
        this.playing = false;
        if (this.onEnd) this.onEnd();
      } else {
        this.setTime(next);
      }
    }
    this._stepPulses(dt);
    this._stepZoom(dt);
    if (this.hovered && this.hoverRing.visible) {
      this.hovered.getWorldPosition(this.hoverRing.position);
      this.hoverRing.quaternion.copy(this.camera.quaternion);
      if (!this.hovered.visible || this.hovered.material.opacity <= 0.06) {
        this._setHover(null);
      }
    }
    this.controls.update();
    this._clampView();
    this.composer.render();
    requestAnimationFrame(() => this._loop());
  }
}
