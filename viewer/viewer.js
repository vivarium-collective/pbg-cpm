// pbg-cpm viewer — renders exported Cellular Potts time-series.
// 2D models render as a top-down cell-field texture; 3D models render as
// instanced boundary voxels with orbit controls. Cells are colored by type
// with a stable per-cell shade so individual cells and domains stay legible.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (s) => document.querySelector(s);
const wrap = $("#canvas-wrap");
const infoEl = $("#info");
const loadingEl = $("#loading");
const scrub = $("#scrub");
const playBtn = $("#play");
const frameLabel = $("#framelabel");

// ---- renderer / scene (persistent) ----
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
wrap.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

let camera, controls, current = null, playing = false, frameIdx = 0;
let lastStep = 0;

// ---- color: type hue + stable per-cell lightness ----
const TYPE_HUE = { 1: 0.58, 2: 0.09, 3: 0.33, 4: 0.78 }; // blue, orange, green, violet
function cellColor(cellId, cellType, out) {
  const hue = TYPE_HUE[cellType] ?? 0.0;
  // deterministic per-cell jitter so neighboring same-type cells are distinct
  const j = ((cellId * 2654435761) >>> 0) / 4294967295;
  const light = 0.42 + 0.30 * j;
  const sat = 0.55 + 0.20 * (1 - j);
  out.setHSL(hue, sat, light);
  return out;
}

// viridis-ish ramp for scalar fields (t in 0..1) -> [r,g,b] bytes
function heat(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [13, 17, 23], [40, 40, 110], [40, 100, 140],
    [30, 150, 130], [120, 195, 70], [250, 232, 60],
  ];
  const f = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(f));
  const a = f - i, s0 = stops[i], s1 = stops[i + 1];
  return [
    (s0[0] + (s1[0] - s0[0]) * a) | 0,
    (s0[1] + (s1[1] - s0[1]) * a) | 0,
    (s0[2] + (s1[2] - s0[2]) * a) | 0,
  ];
}

function buildPalette(cellTypes) {
  // rgba bytes per cell id; medium (0) -> transparent
  const n = cellTypes.length;
  const pal = new Uint8Array(n * 4);
  const c = new THREE.Color();
  for (let id = 1; id < n; id++) {
    cellColor(id, cellTypes[id], c);
    pal[id * 4] = Math.round(c.r * 255);
    pal[id * 4 + 1] = Math.round(c.g * 255);
    pal[id * 4 + 2] = Math.round(c.b * 255);
    pal[id * 4 + 3] = 255;
  }
  return pal;
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
  const d = new THREE.DirectionalLight(0xffffff, 0.9);
  d.position.set(0.6, 1, 0.8);
  scene.add(d);
  const d2 = new THREE.DirectionalLight(0x88aaff, 0.35);
  d2.position.set(-0.7, -0.3, -0.6);
  scene.add(d2);
  scene.userData.lit = true;
}

// ---------- 2D rendering ----------
function setup2D(model) {
  const [nx, ny] = model.dims;
  const pal = buildPalette(model.cell_types);
  const tex = new THREE.DataTexture(new Uint8Array(nx * ny * 4), nx, ny,
    THREE.RGBAFormat);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  const aspect = nx / ny;
  const geo = new THREE.PlaneGeometry(aspect, 1);
  const mat = new THREE.MeshBasicMaterial({ map: tex });
  const plane = new THREE.Mesh(geo, mat);
  scene.add(plane);

  const cam = new THREE.OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0.1, 10);
  cam.position.z = 2;
  camera = cam;
  controls?.dispose?.();
  controls = null;

  const fieldMax = model.field_max || 0;
  current = {
    model, kind: "2d", tex, pal, nx, ny, plane,
    render(fi) {
      const frame = model.frames[fi];
      const labels = frame.labels;
      const field = frame.field;   // optional scalar field (chemotaxis demo)
      const data = tex.image.data;
      for (let i = 0; i < labels.length; i++) {
        const id = labels[i];
        const s = id * 4;
        // flip Y so origin is bottom-left visually
        const x = i % nx, y = ny - 1 - ((i / nx) | 0);
        const d = (y * nx + x) * 4;
        if (id === 0) {
          if (field && fieldMax > 0) {
            const c = heat(field[i] / fieldMax);
            data[d] = c[0]; data[d+1] = c[1]; data[d+2] = c[2]; data[d+3] = 255;
          } else { data[d] = 13; data[d+1] = 17; data[d+2] = 23; data[d+3] = 255; }
        } else { data[d] = pal[s]; data[d+1] = pal[s+1]; data[d+2] = pal[s+2]; data[d+3] = 255; }
      }
      tex.needsUpdate = true;
    },
  };
  fitOrtho();
}

function fitOrtho() {
  if (!current || current.kind !== "2d") return;
  const w = wrap.clientWidth, h = wrap.clientHeight;
  const aspect = current.nx / current.ny;
  const viewAspect = w / h;
  const cam = camera;
  const pad = 1.08;
  if (viewAspect > aspect) {
    cam.top = 0.5 * pad; cam.bottom = -0.5 * pad;
    cam.left = -0.5 * pad * viewAspect; cam.right = 0.5 * pad * viewAspect;
  } else {
    cam.left = -0.5 * pad * aspect / viewAspect * viewAspect; // keep width
    const halfW = 0.5 * pad * aspect;
    cam.left = -halfW; cam.right = halfW;
    cam.top = halfW / viewAspect; cam.bottom = -halfW / viewAspect;
  }
  cam.updateProjectionMatrix();
}

// ---------- 3D rendering ----------
function setup3D(model) {
  const [nx, ny, nz] = model.dims;
  ensureLights();
  const pal = buildPalette(model.cell_types);

  // max instances across frames
  let maxV = 0;
  for (const f of model.frames) maxV = Math.max(maxV, f.voxels.length);
  const geo = new THREE.BoxGeometry(1, 1, 1);
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0.05 });
  const mesh = new THREE.InstancedMesh(geo, mat, maxV);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(maxV * 3), 3);
  mesh.count = 0;
  scene.add(mesh);

  const cam = new THREE.PerspectiveCamera(45, wrap.clientWidth / wrap.clientHeight, 0.1, 5000);
  const c = Math.max(nx, ny, nz);
  cam.position.set(c * 1.4, c * 1.1, c * 1.6);
  camera = cam;
  controls?.dispose?.();
  controls = new OrbitControls(cam, renderer.domElement);
  controls.target.set(nx / 2, ny / 2, nz / 2);
  controls.enableDamping = true;
  controls.update();

  const tmp = new THREE.Object3D();
  const col = new THREE.Color();
  current = {
    model, kind: "3d", mesh, pal,
    render(fi) {
      const vox = model.frames[fi].voxels;
      mesh.count = vox.length;
      for (let i = 0; i < vox.length; i++) {
        const v = vox[i];
        tmp.position.set(v[0], v[1], v[2]);
        tmp.updateMatrix();
        mesh.setMatrixAt(i, tmp.matrix);
        const s = v[3] * 4;
        col.setRGB(pal[s] / 255, pal[s + 1] / 255, pal[s + 2] / 255);
        mesh.setColorAt(i, col);
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    },
  };
}

// ---------- model loading ----------
async function loadModel(entry) {
  loadingEl.style.display = "flex";
  loadingEl.textContent = `loading ${entry.name}…`;
  const res = await fetch("./data/" + entry.file);
  const model = await res.json();
  clearScene();
  playing = false; playBtn.textContent = "▶ Play";
  if (model.is3d) setup3D(model); else setup2D(model);
  frameIdx = 0;
  scrub.max = String(model.frames.length - 1);
  scrub.value = "0";
  current.render(0);
  updateFrameLabel();
  const blurb = {
    cellsort: "CC3D cellsort — differential-adhesion sorting (Steinberg). Condensing cells " +
      "(cohesive) segregate into a cluster engulfed by NonCondensing.",
    chemotaxis: "CC3D bacterium_macrophage — the Bacterium secretes a diffusing attractant " +
      "(ATTR, shown as the heatmap); the Macrophage chemotaxes up the gradient and hunts it.",
    growth: "CC3D cell growth & division — cells grow (target volume ↑) and divide at ~2× " +
      "via mitosis, forming a proliferating colony.",
  }[entry.kind] || "";
  const checks = (entry.checks || []).map((c) =>
    `<div style="font-size:12px;margin-top:3px;color:${c.pass ? '#5ad17f' : '#ff6b6b'}">` +
    `${c.pass ? '✓' : '✗'} ${c.text}</div>`).join("");
  const badge = entry.validated
    ? `<span style="color:#5ad17f">✓ matches CC3D expected behavior</span>`
    : `<span style="color:#ff6b6b">✗ validation failed</span>`;
  infoEl.innerHTML = `<h2>${model.name}</h2><p>${blurb}</p>` +
    `<p style="margin-top:6px;color:#6f7d8c">${model.n_cells} cells · ` +
    `${model.dims.join("×")} lattice · ${model.frames.length} frames</p>` +
    `<div style="margin-top:8px;font-size:12.5px;font-weight:600">${badge}</div>${checks}`;
  loadingEl.style.display = "none";
  onResize();
}

function updateFrameLabel() {
  const f = current.model.frames[frameIdx];
  frameLabel.textContent = `frame ${frameIdx}/${current.model.frames.length - 1} · ${f.mcs} MCS`;
}

function showFrame(i) {
  frameIdx = Math.max(0, Math.min(current.model.frames.length - 1, i | 0));
  current.render(frameIdx);
  scrub.value = String(frameIdx);
  updateFrameLabel();
}

// ---------- ui ----------
scrub.addEventListener("input", () => { playing = false; playBtn.textContent = "▶ Play"; showFrame(+scrub.value); });
playBtn.addEventListener("click", () => {
  if (!current) return;
  playing = !playing;
  playBtn.textContent = playing ? "❚❚ Pause" : "▶ Play";
  if (playing && frameIdx >= current.model.frames.length - 1) showFrame(0);
});

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
    if (t - lastStep > 260) { // ~4 fps playback
      lastStep = t;
      if (frameIdx < current.model.frames.length - 1) showFrame(frameIdx + 1);
      else { playing = false; playBtn.textContent = "▶ Play"; }
    }
  }
  controls?.update?.();
  if (camera) renderer.render(scene, camera);
}
requestAnimationFrame(animate);

// ---------- boot ----------
(async function () {
  const res = await fetch("./data/index.json");
  const { models } = await res.json();
  const ul = $("#models");
  ul.innerHTML = "";
  models.forEach((m, i) => {
    const li = document.createElement("li");
    const vbadge = m.validated === undefined ? "" :
      `<span style="color:${m.validated ? '#5ad17f' : '#ff6b6b'}">${m.validated ? '✓' : '✗'}</span> `;
    li.innerHTML = `<div class="mname">${vbadge}${m.name}</div>` +
      `<div class="mmeta">${m.is3d ? "3D" : "2D"} · ${m.n_cells} cells · ${m.dims.join("×")}</div>`;
    li.addEventListener("click", () => {
      [...ul.children].forEach((c) => c.classList.remove("active"));
      li.classList.add("active");
      loadModel(m);
    });
    ul.appendChild(li);
  });
  loadingEl.textContent = "select a model →";
  // auto-open the first model
  if (models.length) ul.children[0].click();
})();
