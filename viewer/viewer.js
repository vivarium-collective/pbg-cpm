// pbg-cpm viewer — renders exported Cellular Potts time-series with an
// interrogation UI: color-by modes, per-type toggles, hover cell inspection,
// full playback transport, chemical-field overlay, and live measurements.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (s) => document.querySelector(s);
const wrap = $("#canvas-wrap");
const infoEl = $("#info");
const loadingEl = $("#loading");
const tip = $("#hover-tip");
const scrub = $("#scrub");
const playBtn = $("#play");
const speedEl = $("#speed");
const loopEl = $("#loop");
const frameLabel = $("#framelabel");
const colormodeEl = $("#colormode");
const fieldControls = $("#field-controls");
const showFieldEl = $("#showfield");
const fieldOpEl = $("#fieldop");
const legendSection = $("#legend-section");
const legendEl = $("#legend");
const statsEl = $("#stats");
const sparkCanvas = $("#spark");
const sparkLabel = $("#sparklabel");
const validSection = $("#valid-section");
const checksEl = $("#checks");

// ---- renderer / scene (persistent) ----
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
wrap.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f14);
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();

let camera, controls, current = null, playing = false, frameIdx = 0;
let lastStep = 0;
let colorMode = "type";
let hiddenTypes = new Set();
let hoveredCell = 0;      // cell id currently under the cursor (0 = none)

// ---- color helpers ----
const HUES = [0.58, 0.09, 0.33, 0.78, 0.50, 0.00, 0.16, 0.66, 0.42, 0.88, 0.25, 0.72];
const typeHue = (t) => HUES[(Math.max(1, t) - 1) % HUES.length];
function typeSwatch(t) {
  const c = new THREE.Color(); c.setHSL(typeHue(t), 0.6, 0.55);
  return `rgb(${(c.r*255)|0},${(c.g*255)|0},${(c.b*255)|0})`;
}
function heat(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[13,17,23],[40,40,110],[40,100,140],[30,150,130],[120,195,70],[250,232,60]];
  const f = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(f));
  const a = f - i, s0 = stops[i], s1 = stops[i + 1];
  return [(s0[0]+(s1[0]-s0[0])*a)|0, (s0[1]+(s1[1]-s0[1])*a)|0, (s0[2]+(s1[2]-s0[2])*a)|0];
}

// Static palettes: flat-per-type and jittered-per-cell (rgb bytes, index = cellId*3)
function buildPalettes(cellTypes) {
  const n = cellTypes.length;
  const flat = new Uint8Array(n * 3), jit = new Uint8Array(n * 3);
  const c = new THREE.Color();
  for (let id = 1; id < n; id++) {
    const hue = typeHue(cellTypes[id]);
    c.setHSL(hue, 0.62, 0.55);
    flat[id*3] = c.r*255; flat[id*3+1] = c.g*255; flat[id*3+2] = c.b*255;
    const j = ((id * 2654435761) >>> 0) / 4294967295;
    c.setHSL(hue, 0.55 + 0.20*(1-j), 0.42 + 0.30*j);
    jit[id*3] = c.r*255; jit[id*3+1] = c.g*255; jit[id*3+2] = c.b*255;
  }
  return { flat, jit };
}

// color for a cell id at the current frame, honoring color mode + volume ramp
function cellRGB(id, out) {
  const c = current;
  if (colorMode === "cell") {
    out[0] = c.pal.jit[id*3]; out[1] = c.pal.jit[id*3+1]; out[2] = c.pal.jit[id*3+2];
  } else if (colorMode === "volume" && c.volNorm) {
    const h = heat(c.volNorm[id] || 0);
    out[0] = h[0]; out[1] = h[1]; out[2] = h[2];
  } else {
    out[0] = c.pal.flat[id*3]; out[1] = c.pal.flat[id*3+1]; out[2] = c.pal.flat[id*3+2];
  }
}

function clearScene() {
  for (let i = scene.children.length - 1; i >= 0; i--) {
    const o = scene.children[i];
    if (o.isLight) continue;
    scene.remove(o);
    o.geometry?.dispose?.();
    o.material?.dispose?.();
    if (o.isMesh && o.material?.map) o.material.map.dispose?.();
  }
}
function ensureLights() {
  if (scene.userData.lit) return;
  scene.add(new THREE.AmbientLight(0xffffff, 0.62));
  const d = new THREE.DirectionalLight(0xffffff, 0.9); d.position.set(0.6, 1, 0.8); scene.add(d);
  const d2 = new THREE.DirectionalLight(0x88aaff, 0.35); d2.position.set(-0.7, -0.3, -0.6); scene.add(d2);
  scene.userData.lit = true;
}

// ---- per-frame cell volumes (2D: exact pixel counts; drives volume mode + hover + stats) ----
function volsForFrame(fi) {
  const c = current;
  if (c.volCache[fi]) return c.volCache[fi];
  const n = c.model.cell_types.length;
  const counts = new Float32Array(n);
  if (c.kind === "2d") {
    const labels = c.model.frames[fi].labels;
    for (let i = 0; i < labels.length; i++) counts[labels[i]]++;
  } else {
    // boundary voxels only — a surface proxy, not true volume
    const vox = c.model.frames[fi].voxels;
    for (let i = 0; i < vox.length; i++) counts[vox[i][3]]++;
  }
  c.volCache[fi] = counts;
  return counts;
}
// normalize non-medium volumes to 0..1 for the heat ramp (min..max across live cells)
function computeVolNorm(fi) {
  const counts = volsForFrame(fi);
  let mn = Infinity, mx = -Infinity;
  for (let id = 1; id < counts.length; id++) if (counts[id] > 0) {
    if (counts[id] < mn) mn = counts[id];
    if (counts[id] > mx) mx = counts[id];
  }
  const norm = new Float32Array(counts.length);
  const span = mx - mn || 1;
  for (let id = 1; id < counts.length; id++) norm[id] = counts[id] > 0 ? (counts[id]-mn)/span : 0;
  current.volNorm = norm;
}

// ---------- 2D ----------
function setup2D(model) {
  const [nx, ny] = model.dims;
  const tex = new THREE.DataTexture(new Uint8Array(nx*ny*4), nx, ny, THREE.RGBAFormat);
  tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.NearestFilter;
  const aspect = nx / ny;
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(aspect, 1),
    new THREE.MeshBasicMaterial({ map: tex }));
  scene.add(plane);
  const cam = new THREE.OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0.1, 10);
  cam.position.z = 2; camera = cam;
  controls?.dispose?.(); controls = null;

  let fieldMax = model.field_max || 0;
  if (!fieldMax && model.frames[0].field) {
    for (const f of model.frames) for (const v of f.field) if (v > fieldMax) fieldMax = v;
  }
  current = {
    model, kind: "2d", tex, nx, ny, plane, hasField: !!model.frames[0].field, fieldMax,
    pal: buildPalettes(model.cell_types), volCache: {}, volNorm: null,
    render(fi) {
      if (colorMode === "volume") computeVolNorm(fi);
      const frame = model.frames[fi];
      const labels = frame.labels, field = frame.field;
      const data = tex.image.data;
      const showField = this.hasField && showFieldEl.checked;
      const op = (+fieldOpEl.value) / 100;
      const rgb = [0,0,0];
      for (let i = 0; i < labels.length; i++) {
        const id = labels[i];
        const x = i % nx, y = ny - 1 - ((i / nx) | 0);
        const d = (y * nx + x) * 4;
        const hidden = id !== 0 && hiddenTypes.has(model.cell_types[id]);
        if (id === 0 || hidden) {
          if (showField && field && this.fieldMax > 0) {
            const c = heat((field[i] / this.fieldMax) * op);
            data[d]=c[0]; data[d+1]=c[1]; data[d+2]=c[2];
          } else { data[d]=11; data[d+1]=15; data[d+2]=20; }
          data[d+3] = 255;
        } else {
          cellRGB(id, rgb);
          if (id === hoveredCell) { rgb[0]=255; rgb[1]=255; rgb[2]=255; }
          data[d]=rgb[0]; data[d+1]=rgb[1]; data[d+2]=rgb[2]; data[d+3]=255;
        }
      }
      tex.needsUpdate = true;
    },
    // screen uv (0..1) -> cell id at current frame
    pick(uv) {
      const col = Math.min(nx-1, Math.max(0, Math.floor(uv.x * nx)));
      const row = Math.min(ny-1, Math.max(0, Math.floor(uv.y * ny)));
      const idx = (ny - 1 - row) * nx + col;
      return model.frames[frameIdx].labels[idx];
    },
  };
  fitOrtho();
}

function fitOrtho() {
  if (!current || current.kind !== "2d") return;
  const w = wrap.clientWidth, h = wrap.clientHeight;
  const aspect = current.nx / current.ny, viewAspect = w / h, cam = camera, pad = 1.08;
  if (viewAspect > aspect) {
    cam.top = 0.5*pad; cam.bottom = -0.5*pad;
    cam.left = -0.5*pad*viewAspect; cam.right = 0.5*pad*viewAspect;
  } else {
    const halfW = 0.5*pad*aspect;
    cam.left = -halfW; cam.right = halfW;
    cam.top = halfW/viewAspect; cam.bottom = -halfW/viewAspect;
  }
  cam.updateProjectionMatrix();
}

// ---------- 3D ----------
function setup3D(model) {
  const [nx, ny, nz] = model.dims;
  ensureLights();
  let maxV = 0;
  for (const f of model.frames) maxV = Math.max(maxV, f.voxels.length);
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1),
    new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0.05 }), maxV);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(maxV*3), 3);
  mesh.count = 0; scene.add(mesh);

  const cam = new THREE.PerspectiveCamera(45, wrap.clientWidth/wrap.clientHeight, 0.1, 5000);
  const c = Math.max(nx, ny, nz);
  cam.position.set(c*1.4, c*1.1, c*1.6); camera = cam;
  controls?.dispose?.();
  controls = new OrbitControls(cam, renderer.domElement);
  controls.target.set(nx/2, ny/2, nz/2); controls.enableDamping = true; controls.update();

  const tmp = new THREE.Object3D(), col = new THREE.Color();
  current = {
    model, kind: "3d", mesh, hasField: false, fieldMax: 0,
    pal: buildPalettes(model.cell_types), volCache: {}, volNorm: null,
    render(fi) {
      if (colorMode === "volume") computeVolNorm(fi);
      const vox = model.frames[fi].voxels;
      let n = 0; const rgb = [0,0,0];
      for (let i = 0; i < vox.length; i++) {
        const v = vox[i], id = v[3];
        if (hiddenTypes.has(model.cell_types[id])) continue;
        tmp.position.set(v[0], v[1], v[2]); tmp.updateMatrix();
        mesh.setMatrixAt(n, tmp.matrix);
        if (id === hoveredCell) col.setRGB(1,1,1);
        else { cellRGB(id, rgb); col.setRGB(rgb[0]/255, rgb[1]/255, rgb[2]/255); }
        mesh.setColorAt(n, col);
        n++;
      }
      mesh.count = n;
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    },
    pickInstance(instanceId) {
      // instances are packed skipping hidden cells; rebuild the mapping is costly,
      // so pick against the raw voxel list by re-deriving visible order.
      const vox = model.frames[frameIdx].voxels;
      let n = 0;
      for (let i = 0; i < vox.length; i++) {
        const id = vox[i][3];
        if (hiddenTypes.has(model.cell_types[id])) continue;
        if (n === instanceId) return id;
        n++;
      }
      return 0;
    },
  };
}

// ---------- measurements ----------
function distinctCells(fi) {
  const counts = volsForFrame(fi);
  let n = 0;
  for (let id = 1; id < counts.length; id++) if (counts[id] > 0) n++;
  return n;
}
function computeSpark() {
  const m = current.model;
  const arr = new Array(m.frames.length);
  for (let i = 0; i < m.frames.length; i++) arr[i] = distinctCells(i);
  current.spark = arr;
}
function drawSpark() {
  const arr = current.spark; if (!arr) return;
  const cv = sparkCanvas, ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const mx = Math.max(...arr), mn = Math.min(...arr);
  const span = mx - mn || 1, pad = 4;
  ctx.strokeStyle = "#4aa3ff"; ctx.lineWidth = 1.5; ctx.beginPath();
  arr.forEach((v, i) => {
    const x = pad + (W - 2*pad) * (arr.length > 1 ? i/(arr.length-1) : 0);
    const y = H - pad - (H - 2*pad) * (v - mn) / span;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  // current-frame marker
  const cx = pad + (W - 2*pad) * (arr.length > 1 ? frameIdx/(arr.length-1) : 0);
  ctx.fillStyle = "#e6edf3";
  const cy = H - pad - (H - 2*pad) * (arr[frameIdx] - mn) / span;
  ctx.beginPath(); ctx.arc(cx, cy, 2.6, 0, 7); ctx.fill();
}
function updateStats() {
  const m = current.model;
  const counts = volsForFrame(frameIdx);
  const nActive = distinctCells(frameIdx);
  let sum = 0, live = 0;
  for (let id = 1; id < counts.length; id++) if (counts[id] > 0) { sum += counts[id]; live++; }
  const meanVol = live ? (sum/live) : 0;
  const rows = [["active cells", nActive.toLocaleString()]];
  if (current.kind === "2d") rows.push(["mean cell area", `${meanVol.toFixed(0)} px`]);
  else rows.push(["mean surface", `${meanVol.toFixed(0)} vox`]);
  if (m.throughput_mcs_s) {
    rows.push(["throughput", `${m.throughput_mcs_s} MCS/s`]);
    if (m.attempts_per_s) rows.push(["copy-attempts", `${m.attempts_per_s} M/s`]);
  }
  statsEl.innerHTML = rows.map(([k,v]) =>
    `<div class="stat"><span>${k}</span><span>${v}</span></div>`).join("");
  sparkLabel.textContent = current.spark
    ? `active cells over ${current.spark.length} frames` : "";
  drawSpark();
}

// ---------- legend (per-type toggle) ----------
function buildLegend(model) {
  const types = [...new Set(model.cell_types.slice(1))].sort((a,b)=>a-b);
  if (types.length <= 1 && !model.type_names) { legendSection.style.display = "none"; return; }
  legendSection.style.display = "";
  const names = model.type_names;   // index-aligned to type id (imaging models)
  const counts = volsForFrame(0);
  const perType = {};
  for (let id = 1; id < model.cell_types.length; id++)
    if (counts[id] > 0) perType[model.cell_types[id]] = (perType[model.cell_types[id]]||0)+1;
  legendEl.innerHTML = "";
  types.forEach((t) => {
    const nm = names && names[t] ? names[t] : `type ${t}`;
    const row = document.createElement("div");
    row.className = "legend-item" + (hiddenTypes.has(t) ? " off" : "");
    row.innerHTML = `<span class="sw" style="background:${typeSwatch(t)}"></span>` +
      `<span>${nm}</span><span class="cnt">${perType[t]||0}</span>`;
    row.addEventListener("click", () => {
      if (hiddenTypes.has(t)) hiddenTypes.delete(t); else hiddenTypes.add(t);
      row.classList.toggle("off");
      current.render(frameIdx);
    });
    legendEl.appendChild(row);
  });
}

// ---------- model loading ----------
const BLURB = {
  cellsort: "CC3D cellsort — differential-adhesion sorting (Steinberg). Cohesive Condensing " +
    "cells segregate into a cluster engulfed by NonCondensing cells.",
  chemotaxis: "CC3D bacterium_macrophage — the Bacterium secretes a diffusing attractant " +
    "(shown as the heatmap); the Macrophage chemotaxes up the gradient and hunts it.",
  growth: "Cell growth & division — cells grow (target volume ↑) and divide near 2× via " +
    "mitosis, forming a proliferating colony/spheroid.",
  scale: "Scale test — a densely packed lattice showing how many cells the single-threaded " +
    "Rust core sustains, with measured throughput.",
  imaging: "Initialized from REAL imaging — a MIBI-TOF human colon-carcinoma field of view " +
    "(scverse/squidpy). Each cell is placed at its exact segmented pixels, colored by its " +
    "annotated type, then relaxed as a Cellular Potts tissue.",
  ftu: "Converted from a Human Reference Atlas 2D Functional Tissue Unit illustration — the " +
    "colonic-crypt FTU (hubmapconsortium ccf-2d-reference-object-library). Every cell polygon " +
    "in the atlas Crosswalk layer is rasterized and placed with its Cell-Ontology type, then " +
    "relaxed as a Cellular Potts tissue.",
};

async function loadModel(entry) {
  loadingEl.style.display = "flex";
  loadingEl.textContent = `loading ${entry.name}…`;
  const model = await (await fetch("./data/" + entry.file)).json();
  clearScene();
  playing = false; playBtn.textContent = "▶";
  hiddenTypes = new Set(); hoveredCell = 0; colorMode = "type"; colormodeEl.value = "type";
  tip.style.display = "none";
  if (model.is3d) setup3D(model); else setup2D(model);

  // volume color mode is meaningful only for 2D (exact areas)
  colormodeEl.querySelector('option[value="volume"]').disabled = model.is3d;

  fieldControls.style.display = current.hasField ? "" : "none";
  frameIdx = 0;
  scrub.max = String(model.frames.length - 1);
  scrub.value = "0";
  computeSpark();
  buildLegend(model);
  current.render(0);
  updateFrameLabel(); updateStats();

  infoEl.innerHTML = `<h2>${model.name}</h2>` +
    `<div class="desc">${BLURB[entry.kind] || ""}</div>` +
    `<div class="meta">${model.n_cells.toLocaleString()} cells · ${model.dims.join("×")} ` +
    `lattice · ${model.frames.length} frames</div>`;

  if (entry.checks && entry.checks.length) {
    validSection.style.display = "";
    const badge = entry.validated
      ? `<div class="check" style="color:var(--good);font-weight:600">✓ matches expected behavior</div>`
      : `<div class="check" style="color:var(--bad);font-weight:600">✗ validation failed</div>`;
    checksEl.innerHTML = badge + entry.checks.map((c) =>
      `<div class="check" style="color:${c.pass?'var(--good)':'var(--bad)'}">` +
      `${c.pass?'✓':'✗'} ${c.text}</div>`).join("");
  } else validSection.style.display = "none";

  loadingEl.style.display = "none";
  onResize();
}

function updateFrameLabel() {
  const f = current.model.frames[frameIdx];
  frameLabel.textContent = `frame ${frameIdx}/${current.model.frames.length-1} · ${f.mcs} MCS`;
}
function showFrame(i) {
  frameIdx = Math.max(0, Math.min(current.model.frames.length-1, i|0));
  if (colorMode === "volume") current.volNorm = null;   // recompute for the new frame
  current.render(frameIdx);
  scrub.value = String(frameIdx);
  updateFrameLabel(); updateStats();
}

// ---------- transport ----------
function pause() { playing = false; playBtn.textContent = "▶"; }
scrub.addEventListener("input", () => { pause(); showFrame(+scrub.value); });
playBtn.addEventListener("click", () => {
  if (!current) return;
  playing = !playing; playBtn.textContent = playing ? "❚❚" : "▶";
  if (playing && frameIdx >= current.model.frames.length-1) showFrame(0);
});
$("#first").addEventListener("click", () => { pause(); showFrame(0); });
$("#stepback").addEventListener("click", () => { pause(); showFrame(frameIdx-1); });
$("#stepfwd").addEventListener("click", () => { pause(); showFrame(frameIdx+1); });
colormodeEl.addEventListener("change", () => {
  colorMode = colormodeEl.value;
  current.volNorm = null; current.render(frameIdx);
});
showFieldEl.addEventListener("change", () => current.render(frameIdx));
fieldOpEl.addEventListener("input", () => current.render(frameIdx));

// ---------- hover inspection ----------
function setHover(id) {
  if (id === hoveredCell) return;
  hoveredCell = id;
  current.render(frameIdx);
}
wrap.addEventListener("mousemove", (e) => {
  if (!current) return;
  const r = renderer.domElement.getBoundingClientRect();
  ndc.x = ((e.clientX - r.left) / r.width) * 2 - 1;
  ndc.y = -((e.clientY - r.top) / r.height) * 2 + 1;
  raycaster.setFromCamera(ndc, camera);
  let id = 0;
  if (current.kind === "2d") {
    const hit = raycaster.intersectObject(current.plane)[0];
    if (hit && hit.uv) id = current.pick(hit.uv);
  } else {
    const hit = raycaster.intersectObject(current.mesh)[0];
    if (hit && hit.instanceId != null) id = current.pickInstance(hit.instanceId);
  }
  setHover(id);
  if (id === 0) { tip.style.display = "none"; return; }
  const t = current.model.cell_types[id];
  const names = current.model.type_names;
  const tname = names && names[t] ? names[t] : `type ${t}`;
  const vol = volsForFrame(frameIdx)[id] | 0;
  const volLabel = current.kind === "2d" ? `${vol} px` : `${vol} surface vox`;
  tip.innerHTML = `<span class="sw" style="background:${typeSwatch(t)}"></span>` +
    `<b>cell ${id}</b> · ${tname}<br>volume ${volLabel}`;
  tip.style.display = "block";
  const wr = wrap.getBoundingClientRect();
  let lx = e.clientX - wr.left + 14, ly = e.clientY - wr.top + 14;
  if (lx + 160 > wr.width) lx = e.clientX - wr.left - 160;
  tip.style.left = lx + "px"; tip.style.top = ly + "px";
});
wrap.addEventListener("mouseleave", () => { tip.style.display = "none"; setHover(0); });

// ---------- loop ----------
function onResize() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h);
  if (!current) return;
  if (current.kind === "2d") fitOrtho();
  else { camera.aspect = w / h; camera.updateProjectionMatrix(); }
}
window.addEventListener("resize", onResize);

function animate(t) {
  requestAnimationFrame(animate);
  if (playing && current) {
    const interval = 1000 / (+speedEl.value);
    if (t - lastStep > interval) {
      lastStep = t;
      if (frameIdx < current.model.frames.length-1) showFrame(frameIdx+1);
      else if (loopEl.checked) showFrame(0);
      else pause();
    }
  }
  controls?.update?.();
  if (camera) renderer.render(scene, camera);
}
requestAnimationFrame(animate);

// ---------- boot ----------
(async function () {
  const { models } = await (await fetch("./data/index.json")).json();
  const ul = $("#models");
  ul.innerHTML = "";
  models.forEach((m) => {
    const li = document.createElement("li");
    const vb = m.validated === undefined ? "" :
      `<span style="color:${m.validated?'#5ad17f':'#ff6b6b'}">${m.validated?'✓':'✗'}</span> `;
    li.innerHTML = `<div class="mname">${vb}${m.name}</div>` +
      `<div class="mmeta">${m.is3d?"3D":"2D"} · ${(m.n_cells||0).toLocaleString()} cells · ${m.dims.join("×")}</div>`;
    li.addEventListener("click", () => {
      [...ul.children].forEach((c) => c.classList.remove("active"));
      li.classList.add("active");
      loadModel(m);
    });
    ul.appendChild(li);
  });
  loadingEl.textContent = "select a model →";
  if (models.length) ul.children[0].click();
})();
