import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';
import URDFLoader from 'urdf-loader';

// ---------------- profiles ----------------
const PROFILES = await (await fetch('data/profiles.json', { cache: 'no-store' })).json();
let profileName = localStorage.getItem('dz_profile') || 'deepcybo';
if (!PROFILES[profileName]) profileName = Object.keys(PROFILES)[0];
let P = PROFILES[profileName];

let samples = [];
let runs = {};
let smoothMode = localStorage.getItem('dz_smooth') || 'none';
let runName = null;
let cur = 0, t = 0, playing = false;
let replayData = null;   // built for replay profiles
let planBot = null, planOn = false, planAhead = 16;

// ---------------- three.js scene ----------------
const canvas = document.getElementById('three');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111318);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 30);
camera.position.set(1.1, 0.7, 1.1);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.25, 0);
scene.add(new THREE.HemisphereLight(0xffffff, 0x334, 1.1));
const dl = new THREE.DirectionalLight(0xffffff, 1.2);
dl.position.set(2, 3, 2);
scene.add(dl);
scene.add(new THREE.GridHelper(2, 20, 0x2a3040, 0x1c212c));

let camMode = false;
const IMG_ASPECT = 4 / 3;

// ---------------- fisheye post-process ----------------
// The real head/wrist lenses are wide equidistant fisheyes (~147° H head).
// A plain pinhole render only matches the footage near the image center.
// Fix: render the robot with a WIDE pinhole into an off-screen target, then
// resample through the calibrated fisheye model (Kannala-Brandt r=f*theta_d,
// theta_d = theta*(1+k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8)) so the overlay
// bows exactly like the lens, out to the corners where the arms sit.
let fisheyeOn = true;
const FISHEYE_RENDER_FOV = 150;   // deg, vertical FOV of the off-screen pinhole pass
const FISHEYE_SS = 2.0;           // supersample factor for the render target
// Head-cam ChArUco calibration (equidistant, resolution-independent coeffs).
const FISHEYE_K = { head: [-0.027929, 0.012395, -0.010687, 0.002635],
                    wrist: [0, 0, 0, 0] };   // no wrist calib; pure equidistant
const fisheyeRT = new THREE.WebGLRenderTarget(2, 2, {
  minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
  format: THREE.RGBAFormat, depthBuffer: true });
const fisheyeScene = new THREE.Scene();
const fisheyeCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const fisheyeMat = new THREE.ShaderMaterial({
  transparent: true,
  uniforms: {
    uRT: { value: fisheyeRT.texture },
    uAspect: { value: IMG_ASPECT },
    uThetaEdge: { value: THREE.MathUtils.degToRad(80) / 2 },
    uThetaMax: { value: THREE.MathUtils.degToRad(85) },
    uRenderTan: { value: Math.tan(THREE.MathUtils.degToRad(FISHEYE_RENDER_FOV) / 2) },
    uK: { value: new THREE.Vector4(...FISHEYE_K.head) },
  },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
  fragmentShader: `
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uRT;
    uniform float uAspect, uThetaEdge, uThetaMax, uRenderTan;
    uniform vec4 uK;
    void main() {
      float ox = (vUv.x - 0.5) * 2.0 * uAspect;
      float oy = (vUv.y - 0.5) * 2.0;
      float rho = sqrt(ox * ox + oy * oy);
      if (rho < 1e-6) { gl_FragColor = texture2D(uRT, vec2(0.5)); return; }
      float thetaD = rho * uThetaEdge;              // distorted angle at this pixel
      float th = thetaD;                            // invert the KB polynomial
      for (int i = 0; i < 6; i++) {
        float t2 = th * th;
        float poly = 1.0 + uK.x * t2 + uK.y * t2 * t2 + uK.z * t2 * t2 * t2 + uK.w * t2 * t2 * t2 * t2;
        th = thetaD / poly;
      }
      if (th > uThetaMax) { gl_FragColor = vec4(0.0); return; }
      float r = tan(th);                            // pinhole-normalized ray radius
      vec2 ab = r * vec2(ox, oy) / rho;             // (a,b) = (X/Z, Y/Z) ray
      vec2 rtUv = 0.5 + 0.5 * vec2(ab.x / (uRenderTan * uAspect), ab.y / uRenderTan);
      if (rtUv.x < 0.0 || rtUv.x > 1.0 || rtUv.y < 0.0 || rtUv.y > 1.0) { gl_FragColor = vec4(0.0); return; }
      gl_FragColor = texture2D(uRT, rtUv);
    }`,
});
fisheyeScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), fisheyeMat));
function curFov() { return (calib && calib.fov != null) ? calib.fov : +fovSlider.value; }

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = camMode ? IMG_ASPECT : w / h;
    camera.updateProjectionMatrix();
    fisheyeRT.setSize(Math.max(2, Math.round(w * FISHEYE_SS)),
                      Math.max(2, Math.round(h * FISHEYE_SS)));
  }
}

// ---------------- robot loading ----------------
let predBot = null, gtBot = null;

function meshLoaderFor() {
  // IMPORTANT: sub-loaders must share the URDF's LoadingManager, otherwise
  // manager.onLoad fires before meshes arrive and material tinting is skipped.
  return (path, manager, done) => {
    const lower = path.toLowerCase();
    if (lower.endsWith('.fbx')) new FBXLoader(manager).load(path, (o) => done(o), null, (e) => done(null, e));
    else if (lower.endsWith('.stl')) new STLLoader(manager).load(path, (g) => done(new THREE.Mesh(g)), null, (e) => done(null, e));
    else if (lower.endsWith('.dae')) new ColladaLoader(manager).load(path, (c) => done(c.scene), null, (e) => done(null, e));
    else done(null, new Error('unsupported mesh: ' + path));
  };
}

function loadRobot(profile, tint, ghost) {
  return new Promise((res) => {
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    loader.packages = profile.packages;
    loader.loadMeshCb = meshLoaderFor();
    let robot = null, finished = false;
    loader.load(profile.urdf, (r) => { robot = r; });
    const finish = () => {
      if (finished || !robot) return;
      finished = true;
      robot.rotation.x = -Math.PI / 2;
      robot.rotation.z = THREE.MathUtils.degToRad(profile.yaw || 0);  // yaw fix (about vertical)
      // collect first, modify after: adding children DURING traverse() recurses
      // into the additions forever (wireframe-of-wireframe stack overflow)
      const meshes = [];
      robot.traverse((c) => { if (c.isMesh && !c.userData.dzWire) meshes.push(c); });
      for (const c of meshes) {
        const m = new THREE.MeshStandardMaterial({
          color: tint, transparent: ghost, opacity: ghost ? 0.28 : 1.0,
          roughness: 0.55, metalness: 0.15, depthWrite: !ghost,
        });
        m.userData.dzTinted = true;
        c.material = m;
        if (ghost) {
          const wf = new THREE.Mesh(c.geometry, new THREE.MeshBasicMaterial({
            color: tint, wireframe: true, transparent: true, opacity: 0.35 }));
          wf.userData.dzWire = true;
          c.add(wf);
        }
      }
      scene.add(robot);
      // straggler pass: any mesh that arrives after onLoad still gets tinted
      setTimeout(() => {
        const late = [];
        robot.traverse((c) => {
          if (c.isMesh && !c.userData.dzWire &&
              (!c.material || !c.material.userData || !c.material.userData.dzTinted)) late.push(c);
        });
        for (const c of late) {
          const m = new THREE.MeshStandardMaterial({
            color: tint, transparent: ghost, opacity: ghost ? 0.28 : 1.0,
            roughness: 0.55, metalness: 0.15, depthWrite: !ghost });
          m.userData.dzTinted = true;
          c.material = m;
        }
      }, 3000);
      res(robot);
    };
    manager.onLoad = finish;
    manager.onError = () => {};       // missing meshes (e.g. G1 wrists) are non-fatal
    setTimeout(finish, 15000);        // safety net if a mesh never resolves
  });
}

function setPose(robot, vec) {
  if (!robot) return;
  P.dims.forEach((d, i) => {
    if (!d.joint || i >= vec.length) return;
    const j = robot.joints[d.joint];
    if (!j) return;
    let v = vec[i];
    if (d.gripperMax) {
      const upper = (j.limit && Number(j.limit.upper)) || d.gripperMax;
      v = Math.max(0, Math.min(1, v / d.gripperMax)) * upper;
    }
    robot.setJointValue(d.joint, v);
  });
}

// ---------------- data / profile switching ----------------
// ---------------- action smoothing ----------------
function smoothChunk(chunk, mode) {
  if (mode === 'none') return chunk;
  const T = chunk.length, D = chunk[0].length;
  const get = (tt) => chunk[Math.min(Math.max(tt, 0), T - 1)];
  const out = [];
  if (mode === 'b3' || mode === 'b5') {
    const w = mode === 'b3' ? [1, 2, 1] : [1, 4, 6, 4, 1];
    const half = (w.length - 1) / 2, norm = w.reduce((x, y) => x + y, 0);
    for (let tt = 0; tt < T; tt++) {
      const row = new Array(D).fill(0);
      for (let k = 0; k < w.length; k++) {
        const s = get(tt + k - half);
        for (let d = 0; d < D; d++) row[d] += w[k] * s[d];
      }
      out.push(row.map((v) => v / norm));
    }
  } else {                       // causal EMA, y0 = x0 (shows the lag EMA adds)
    const alpha = mode === 'ema3' ? 0.3 : 0.5;
    let prev = null;
    for (let tt = 0; tt < T; tt++) {
      const row = prev === null ? chunk[tt].slice()
        : chunk[tt].map((v, d) => alpha * v + (1 - alpha) * prev[d]);
      out.push(row); prev = row;
    }
  }
  return out;
}

function applySmoothing() {
  for (const rn of Object.keys(runs)) {
    const r = runs[rn];
    r.samples.forEach((sm, i) => {
      if (!sm.pred0) sm.pred0 = sm.pred;
      sm.pred = smoothChunk(sm.pred0, smoothMode);
      const gt = samples[i].gt;
      let tot = 0;
      sm.mse_t = sm.pred.map((row, tt) => {
        let e = 0;
        for (let d = 0; d < row.length; d++) { const df = row[d] - gt[tt][d]; e += df * df; }
        e /= row.length; tot += e; return e;
      });
      sm.mse = tot / sm.pred.length;
      sm.ratio = samples[i].hold_mse / Math.max(sm.mse, 1e-9);
    });
    r.overall_mse = r.samples.reduce((x, s2) => x + s2.mse, 0) / r.samples.length;
    const oh = samples.reduce((x, s2) => x + s2.hold_mse, 0) / samples.length;
    r.overall_ratio = oh / r.overall_mse;
  }
}

function buildReplayData() {
  if (!P.replay) {
    replayData = null;
    document.getElementById('tslider').max = P.horizon - 1;
    return;
  }
  const rs = runs[runName].samples;
  const ord = samples.map((s, i) => i).sort((x, y) => samples[x].idx - samples[y].idx);
  const chunks = ord.map((i) => ({ pred: rs[i].pred, gt: samples[i].gt, idx: samples[i].idx,
                                   ratio: rs[i].ratio }));
  const EX = 8, n = chunks.length, execLen = n * EX;
  const sp = [], sg = [];
  for (let s = 0; s < execLen; s++) {
    const k = Math.min(Math.floor(s / EX), n - 1), o = s - k * EX;
    sp.push(chunks[k].pred[o]); sg.push(chunks[k].gt[o]);
  }
  replayData = { EX, n, execLen, chunks, sp, sg };
  document.getElementById('tslider').max = execLen - 1;
}

async function loadProfile(name) {
  profileName = name; P = PROFILES[name];
  localStorage.setItem('dz_profile', name);
  samples = await (await fetch(P.samplesUrl, { cache: 'no-store' })).json();
  runs = await (await fetch(P.runsUrl, { cache: 'no-store' })).json();
  runName = Object.keys(runs).sort()[0];
  cur = 0; t = 0;
  applySmoothing();
  buildReplayData();
  if (predBot) { scene.remove(predBot); predBot = null; }
  if (gtBot) { scene.remove(gtBot); gtBot = null; }
  if (planBot) { scene.remove(planBot); planBot = null; }
  [predBot, gtBot] = await Promise.all([
    loadRobot(P, 0x7aa2f7, false), loadRobot(P, 0x35e08a, true)]);
  if (P.replay) { planBot = await loadRobot(P, 0xe0a93e, true); planBot.visible = planOn; }
  document.getElementById('planctl').style.display = P.replay ? 'inline' : 'none';
  buildCamSelector(); loadCalib();
  buildRunSelector(); updateStats(); renderList(); select(order()[0]);
}

const profsel = document.getElementById('profsel');
Object.entries(PROFILES).forEach(([k, v]) => profsel.add(new Option(v.label, k)));
profsel.value = profileName;
profsel.onchange = () => loadProfile(profsel.value);

const runsel = document.getElementById('runsel');
function buildRunSelector() {
  runsel.innerHTML = '';
  Object.keys(runs).sort().forEach((r) => runsel.add(new Option(r, r)));
  runsel.value = runName;
}
runsel.onchange = () => { runName = runsel.value; buildReplayData(); renderList(); select(cur); updateStats(); updateDream(); };

const smoothsel = document.getElementById('smoothsel');
smoothsel.value = smoothMode;
smoothsel.onchange = () => {
  smoothMode = smoothsel.value;
  localStorage.setItem('dz_smooth', smoothMode);
  applySmoothing(); buildReplayData();
  updateStats(); renderList(); select(cur);
};

function updateStats() {
  const r = runs[runName];
  const wins = r.samples.filter((s) => s.ratio > 1).length;
  document.getElementById('runstats').innerHTML =
    `overall MSE <b>${r.overall_mse.toFixed(5)}</b> · ratio vs freeze <b>${r.overall_ratio.toFixed(2)}×</b> · beats baseline on <b>${wins}/${r.samples.length}</b>`;
}

// ---------------- sample list ----------------
let sortMode = 'ratio_asc';
document.querySelectorAll('[data-sort]').forEach((b) => b.onclick = () => { sortMode = b.dataset.sort; renderList(); });
function order() {
  const rs = runs[runName].samples;
  const ix = samples.map((_, i) => i);
  if (sortMode === 'ratio_asc') ix.sort((a, b) => rs[a].ratio - rs[b].ratio);
  if (sortMode === 'ratio_desc') ix.sort((a, b) => rs[b].ratio - rs[a].ratio);
  return ix;
}
function chip(ratio) {
  const c = ratio > 1 ? 'var(--good)' : ratio > 0.8 ? 'var(--warn)' : 'var(--bad)';
  return `<span class="chip" style="background:${c};color:#10131a">${ratio.toFixed(2)}×</span>`;
}
function renderList() {
  if (replayData) {
    const r = runs[runName];
    document.getElementById('samplelist').innerHTML = `
      <div class="sample active">
        <div class="top"><span>FULL EPISODE — ${replayData.n} replan points</span>${chip(r.overall_ratio)}</div>
        <div class="prompt">${samples[0].prompt || ''}</div>
      </div>
      <div style="padding:8px 10px;color:var(--dim);font-size:11px">
        Timeline = continuous teacher-forced execution: every 8 steps a new chunk
        takes over (what a real controller would run). Toggle the orange robot to
        see the current plan's unexecuted future.
      </div>`;
    return;
  }
  const rs = runs[runName].samples;
  document.getElementById('samplelist').innerHTML = order().map((i) => `
    <div class="sample ${i === cur ? 'active' : ''}" data-i="${i}">
      <div class="top"><span>#${i} · ep ${samples[i].episode}</span>${chip(rs[i].ratio)}</div>
      <div class="prompt">${samples[i].prompt || '—'}</div>
    </div>`).join('');
  document.querySelectorAll('.sample').forEach((el) => el.onclick = () => select(+el.dataset.i));
}

// ---------------- charts ----------------
function drawLines(cv, series, labels) {
  const ctx = cv.getContext('2d');
  const W = cv.width = cv.clientWidth * 2, H = cv.height = cv.clientHeight * 2;
  ctx.clearRect(0, 0, W, H);
  const all = series.flat();
  const mx = Math.max(...all) * 1.05 || 1, mn = Math.min(0, ...all);
  const n = series[0].length;
  const X = (i) => 30 + (W - 40) * i / (n - 1);
  const Y = (v) => H - 20 - (H - 35) * (v - mn) / (mx - mn);
  ctx.font = '20px sans-serif'; ctx.fillStyle = '#8a93a6';
  ctx.fillText(mx.toPrecision(2), 4, 24);
  const colors = ['#7aa2f7', '#4caf7d', '#e0a93e'];
  series.forEach((s, k) => {
    ctx.strokeStyle = colors[k]; ctx.lineWidth = 3; ctx.beginPath();
    s.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)));
    ctx.stroke();
    if (labels) { ctx.fillStyle = colors[k]; ctx.fillText(labels[k], 40 + k * 220, 24); }
  });
  ctx.strokeStyle = '#ffffff55'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(X(t), 30); ctx.lineTo(X(t), H - 20); ctx.stroke();
}

function renderDetail() {
  const s = samples[cur], r = runs[runName].samples[cur];
  document.getElementById('detail').innerHTML = `
    <img id="detailframe" src="${s.frame || ''}">
    <div style="margin-top:8px;color:var(--dim)">“${s.prompt}”</div>
    <h3>${replayData ? 'metrics (whole episode, executed steps)' : `metrics (horizon 1→${P.horizon})`}</h3>
    <div class="metric-row"><span>model MSE (chunk mean)</span><b>${r.mse.toFixed(5)}</b></div>
    <div class="metric-row"><span>freeze-baseline MSE</span><b>${s.hold_mse.toFixed(5)}</b></div>
    <div class="metric-row"><span>ratio (>1 beats freeze)</span>${chip(r.ratio)}</div>
    <h3>action magnitude by step — mean |value| across dims</h3>
    <canvas class="chart" id="divchart" style="height:120px"></canvas>
    <h3>per-dimension: pred (blue) vs gt (green)</h3>
    <div id="dimgrid">${P.dims.map((d, i) => `
      <div><div style="font-size:10px;color:var(--dim)">${d.name}</div>
      <canvas class="chart dim" data-d="${i}" style="height:56px"></canvas></div>`).join('')}
    </div>`;
  drawCharts();
}
function drawCharts() {
  if (replayData) {
    const dc = document.getElementById('divchart');
    if (dc) {
      const magP = replayData.sp.map((row) => row.reduce((x, v) => x + Math.abs(v), 0) / row.length);
      const magG = replayData.sg.map((row) => row.reduce((x, v) => x + Math.abs(v), 0) / row.length);
      drawLines(dc, [magP, magG], ['|pred executed|', '|gt|']);
    }
    document.querySelectorAll('canvas.dim').forEach((cv) => {
      const d = +cv.dataset.d;
      drawLines(cv, [replayData.sp.map((row) => row[d]), replayData.sg.map((row) => row[d])]);
    });
    return;
  }
  const s = samples[cur], r = runs[runName].samples[cur];
  const dc = document.getElementById('divchart');
  if (dc) {
    const magP = r.pred.map((row) => row.reduce((a, v) => a + Math.abs(v), 0) / row.length);
    const magG = s.gt.map((row) => row.reduce((a, v) => a + Math.abs(v), 0) / row.length);
    drawLines(dc, [magP, magG], ['|pred|', '|gt|']);
  }
  document.querySelectorAll('canvas.dim').forEach((cv) => {
    const d = +cv.dataset.d;
    drawLines(cv, [r.pred.map((row) => row[d]), s.gt.map((row) => row[d])]);
  });
}

// ---------------- playback ----------------
const tslider = document.getElementById('tslider');
function framePath() {
  return `${P.framesSeqBase}/idx${samples[cur].idx}_t${t}.jpg`;
}
function select(i) {
  cur = i; t = 0; tslider.value = 0;
  renderList(); renderDetail(); applyPose();
}
function applyPose() {
  if (replayData) {
    const { EX, n, chunks } = replayData;
    const k = Math.min(Math.floor(t / EX), n - 1), o = t - k * EX;
    setPose(predBot, chunks[k].pred[o]);
    setPose(gtBot, chunks[k].gt[o]);
    if (planBot && planOn) setPose(planBot, chunks[k].pred[Math.min(o + planAhead, chunks[k].pred.length - 1)]);
    const fp = `${P.framesSeqBase}/idx${chunks[k].idx}_t${o}.jpg`;
    const df2 = document.getElementById('detailframe'); if (df2) df2.src = fp;
    const af2 = document.getElementById('arframe'); if (af2 && camMode) af2.src = camFramePath();
    document.getElementById('tlabel').textContent =
      `step ${t}/${replayData.execLen} · chunk ${k} (+${o}) · ratio ${chunks[k].ratio.toFixed(2)}×`;
    updateDream();
    drawCharts();
    return;
  }
  const s = samples[cur], r = runs[runName].samples[cur];
  setPose(predBot, r.pred[t]);
  setPose(gtBot, s.gt[t]);
  const df = document.getElementById('detailframe');
  if (df) df.src = framePath();
  const af = document.getElementById('arframe');
  if (af && camMode) af.src = camFramePath();
  document.getElementById('tlabel').textContent = `t=${t} (${Math.round(t / 30 * 1000)} ms)`;
  updateDream();
  drawCharts();
}
tslider.oninput = () => { t = +tslider.value; applyPose(); };
document.getElementById('play').onclick = (e) => {
  playing = !playing;
  e.target.textContent = playing ? '⏸' : '▶';
};
setInterval(() => {
  if (!playing) return;
  t = (t + 1) % (replayData ? replayData.execLen : P.horizon); tslider.value = t; applyPose();
}, 1000 / 30);

// ---------------- AR cam view (multi-camera) + calibration ----------------
const arframe = document.getElementById('arframe');
const fovSlider = document.getElementById('fov');
const OPT2THREE = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
const savedCam = { pos: new THREE.Vector3(), quat: new THREE.Quaternion() };
const viewerEl = document.getElementById('viewer');
let activeCam = 0;
function camDef() { return (P.cameras && P.cameras[activeCam]) || null; }
function calKey() { return `dz_cam_calib_${profileName}_${camDef() ? camDef().id : 'x'}`; }
let calib = null;   // { dp:[3], dq:[4], fov }
function loadCalib() {
  // localStorage (user's hand-tuned) wins; otherwise fall back to the
  // profile's per-camera default calib (authoritative FOV / offset).
  calib = JSON.parse(localStorage.getItem(calKey()) || 'null');
  if (!calib && camDef() && camDef().calib) calib = JSON.parse(JSON.stringify(camDef().calib));
}

const camselEl = document.getElementById('camsel');
function buildCamSelector() {
  camselEl.innerHTML = '';
  (P.cameras || []).forEach((c, i) => camselEl.add(new Option(c.label, i)));
  activeCam = 0;
  camselEl.style.display = (P.cameras && P.cameras.length > 1) ? 'inline' : 'none';
}
camselEl.onchange = () => {
  activeCam = +camselEl.value;
  loadCalib(); syncSliders();
  if (camMode) { arframe.src = camFramePath(); updateDream(); }
};

function camFramePath() {
  const c = camDef();
  return `${c ? c.framesBase : P.framesSeqBase}/idx${replayData
    ? replayData.chunks[Math.min(Math.floor(t / replayData.EX), replayData.n - 1)].idx
    : samples[cur].idx}_t${replayData ? t % replayData.EX : t}.jpg`;
}

// base pose of the active camera at the CURRENT displayed timestep.
// Link-mounted cameras follow the GT robot's forward kinematics.
const _bp = new THREE.Vector3(), _bq = new THREE.Quaternion();
function camBasePose() {
  const c = camDef();
  if (!c) return null;
  if (c.link && gtBot && gtBot.links && gtBot.links[c.link]) {
    const link = gtBot.links[c.link];
    link.getWorldPosition(_bp);
    link.getWorldQuaternion(_bq);
    _bq.multiply(OPT2THREE);
    return { pos: _bp, quat: _bq };
  }
  if (c.pose) {
    _bp.fromArray(c.pose.pos);
    const m = new THREE.Matrix4().lookAt(_bp,
      new THREE.Vector3().fromArray(c.pose.target), new THREE.Vector3(0, 1, 0));
    _bq.setFromRotationMatrix(m);
    return { pos: _bp, quat: _bq };
  }
  return null;
}

function syncCamView() {
  scene.background = null;
  const base = camBasePose();
  if (!base) return;
  if (calib) {
    const dq = calib.dq ? new THREE.Quaternion().fromArray(calib.dq)
      : new THREE.Quaternion().setFromEuler(new THREE.Euler(...(calib.eul || [0, 0, 0])));
    const dp = new THREE.Vector3().fromArray(calib.dp).applyQuaternion(base.quat);
    camera.position.copy(base.pos).add(dp);
    camera.quaternion.copy(base.quat).multiply(dq);
    camera.fov = calib.fov;
  } else {
    camera.position.copy(base.pos);
    camera.quaternion.copy(base.quat);
    camera.fov = +fovSlider.value;
  }
  camera.aspect = IMG_ASPECT;
  camera.updateProjectionMatrix();
}
fovSlider.oninput = () => { if (camMode && !calib) syncCamView(); };

const fisheyeBtn = document.getElementById('fisheye');
fisheyeBtn.style.background = fisheyeOn ? '#3a4a72' : '';
fisheyeBtn.onclick = () => {
  fisheyeOn = !fisheyeOn;
  fisheyeBtn.style.background = fisheyeOn ? '#3a4a72' : '';
};

document.getElementById('camview').onclick = (e) => {
  if (!P.cameras || !P.cameras.length) { alert('No cameras defined for this profile.'); return; }
  camMode = !camMode;
  e.target.style.background = camMode ? '#3a4a72' : '';
  controls.enabled = !camMode;
  arframe.style.display = camMode ? 'block' : 'none';
  manualPanel.style.display = camMode ? 'block' : 'none';
  if (camMode) {
    loadCalib(); syncSliders();
    savedCam.pos.copy(camera.position); savedCam.quat.copy(camera.quaternion);
    arframe.src = camFramePath();
  }
  updateDream();
  if (!camMode) {
    camera.position.copy(savedCam.pos); camera.quaternion.copy(savedCam.quat);
    camera.fov = 50; camera.updateProjectionMatrix();
    scene.background = new THREE.Color(0x111318);
  }
};

function containRect() {
  const W = viewerEl.clientWidth, H = viewerEl.clientHeight;
  let w = W, h = W / IMG_ASPECT;
  if (h > H) { h = H; w = H * IMG_ASPECT; }
  return { x: (W - w) / 2, y: (H - h) / 2, w, h };
}
function applyCanvasRect() {
  if (camMode) {
    const r = containRect();
    canvas.style.position = 'absolute';
    canvas.style.left = r.x + 'px'; canvas.style.top = r.y + 'px';
    canvas.style.width = r.w + 'px'; canvas.style.height = r.h + 'px';
  } else {
    canvas.style.position = 'relative';
    canvas.style.left = ''; canvas.style.top = '';
    canvas.style.width = '100%'; canvas.style.height = '100%';
  }
}

// ---- calibration: free-form multi-frame clicks -> link-frame offset ----
const G1_KEYS = [
  ['arm_l_end_link', 'LEFT gripper / end of left arm'],
  ['arm_r_end_link', 'RIGHT gripper / end of right arm'],
  ['arm_l_link4', 'LEFT elbow area'],
  ['arm_r_link4', 'RIGHT elbow area'],
  ['head_link2', 'HEAD'],
  ['body_link2', 'CHEST / torso center'],
];
const DC_KEYS = [
  ['left_gripper_finger_left_link', 'LEFT gripper finger tip'],
  ['right_gripper_finger_left_link', 'RIGHT gripper finger tip'],
  ['left_wrist_pitch_link', 'LEFT wrist joint'],
  ['right_wrist_pitch_link', 'RIGHT wrist joint'],
  ['left_elbow_pitch_link', 'LEFT elbow'],
  ['right_elbow_pitch_link', 'RIGHT elbow'],
];
const CALIB_KEYS_BY_PROFILE = {
  deepcybo: DC_KEYS, deepcybo_ep0: DC_KEYS,
  agibot: G1_KEYS, agibot_replay: G1_KEYS,
};
let calibrating = false, calibPts = [];   // {P:[3], u, v, basePos:[3], baseQuat:[4]}
const banner = document.getElementById('calibbanner');
function calibKeys() { return CALIB_KEYS_BY_PROFILE[profileName] || []; }
function calibCount() {
  document.getElementById('calibcount').textContent = `${calibPts.length} pts`;
}
document.getElementById('calib').onclick = () => {
  if (!camMode || !calibKeys().length) { alert('Enter cam view first (profile needs calibration keypoints).'); return; }
  calibrating = true; calibPts = [];
  playing = false;
  predBot.visible = false; gtBot.visible = false;
  const sel = document.getElementById('calibkey');
  sel.innerHTML = '';
  calibKeys().forEach(([link, label]) => sel.add(new Option(label, link)));
  banner.style.display = 'block';
  calibCount();
};
document.getElementById('calibsolve').onclick = () => {
  if (calibPts.length < 4) { alert(`Only ${calibPts.length} points — collect at least 4.`); return; }
  solveCalib(); endCalib(true);
};
document.getElementById('calibundo').onclick = () => { calibPts.pop(); calibCount(); };
document.getElementById('calibcancel').onclick = () => endCalib(false);
viewerEl.addEventListener('click', (ev) => {
  if (!calibrating) return;
  const r = containRect(), vb = viewerEl.getBoundingClientRect();
  const px = (ev.clientX - vb.left - r.x) / r.w, py = (ev.clientY - vb.top - r.y) / r.h;
  if (px < 0 || px > 1 || py < 0 || py > 1) return;
  const key = document.getElementById('calibkey').value;
  const link = gtBot.links[key];
  if (!link) { alert('link not found: ' + key); return; }
  const Pw = new THREE.Vector3();
  link.getWorldPosition(Pw);
  const base = camBasePose();
  if (!base) return;
  calibPts.push({ P: Pw.toArray(), u: px, v: py,
                  basePos: base.pos.toArray(), baseQuat: base.quat.toArray() });
  calibCount();
});
function endCalib(done) {
  calibrating = false;
  banner.style.display = 'none';
  predBot.visible = document.getElementById('showpred').checked;
  gtBot.visible = document.getElementById('showgt').checked;
  if (done && camMode) syncCamView();
}
function projectPt(Pw, pos, quat, fovDeg) {
  const q = new THREE.Quaternion().fromArray(quat).invert();
  const p = new THREE.Vector3().fromArray(Pw).sub(new THREE.Vector3().fromArray(pos)).applyQuaternion(q);
  if (p.z > -1e-4) return null;
  const f = 0.5 / Math.tan(THREE.MathUtils.degToRad(fovDeg) / 2);
  return { u: 0.5 + (p.x / -p.z) * f / IMG_ASPECT, v: 0.5 - (p.y / -p.z) * f };
}
// params = [dx,dy,dz, rx,ry,rz, fov]; per-point camera pose = base_i ∘ offset
function reprojErr(params, seedQ) {
  const [dx, dy, dz, rx, ry, rz, fov] = params;
  const dq = new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz)).premultiply(seedQ);
  let e = 0, n = 0;
  for (const pt of calibPts) {
    const bq = new THREE.Quaternion().fromArray(pt.baseQuat);
    const q = bq.clone().multiply(dq);
    const dp = new THREE.Vector3(dx, dy, dz).applyQuaternion(bq);
    const pos = new THREE.Vector3().fromArray(pt.basePos).add(dp);
    const pr = projectPt(pt.P, pos.toArray(), q.toArray(), fov);
    if (!pr) { e += 4; n++; continue; }
    e += (pr.u - pt.u) ** 2 + (pr.v - pt.v) ** 2; n++;
  }
  return n ? e / n : 1e9;
}
function nelderMead(f, x0, step, iters) {
  const n = x0.length;
  let sim = [x0.slice()];
  for (let i = 0; i < n; i++) { const x = x0.slice(); x[i] += step[i]; sim.push(x); }
  let fv = sim.map(f);
  for (let it = 0; it < iters; it++) {
    const ord = fv.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]).map((p) => p[1]);
    sim = ord.map((i) => sim[i]); fv = ord.map((i) => fv[i]);
    const cen = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) cen[j] += sim[i][j] / n;
    const refl = cen.map((c, j) => 2 * c - sim[n][j]);
    const fr = f(refl);
    if (fr < fv[0]) {
      const ex = cen.map((c, j) => 3 * c - 2 * sim[n][j]);
      const fe = f(ex);
      if (fe < fr) { sim[n] = ex; fv[n] = fe; } else { sim[n] = refl; fv[n] = fr; }
    } else if (fr < fv[n - 1]) { sim[n] = refl; fv[n] = fr; }
    else {
      const con = cen.map((c, j) => 0.5 * (c + sim[n][j]));
      const fc = f(con);
      if (fc < fv[n]) { sim[n] = con; fv[n] = fc; }
      else for (let i = 1; i <= n; i++) { sim[i] = sim[i].map((v, j) => 0.5 * (v + sim[0][j])); fv[i] = f(sim[i]); }
    }
  }
  const best = fv.indexOf(Math.min(...fv));
  return { x: sim[best], f: fv[best] };
}
function solveCalib() {
  const seeds = [
    new THREE.Quaternion(),
    new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI),
    new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2),
    new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2),
    new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI),
  ];
  let best = null;
  for (const s of seeds) {
    const r = nelderMead((p) => reprojErr(p, s),
      [0, 0, 0, 0, 0, 0, +fovSlider.value], [0.04, 0.04, 0.04, 0.12, 0.12, 0.12, 6], 400);
    if (!best || r.f < best.f) best = { ...r, seed: s };
  }
  const [dx, dy, dz, rx, ry, rz, fov] = best.x;
  const dq = new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz)).premultiply(best.seed);
  const eulOut = new THREE.Euler().setFromQuaternion(dq);
  calib = { dp: [dx, dy, dz], eul: [eulOut.x, eulOut.y, eulOut.z],
            fov: Math.max(30, Math.min(90, fov)),
            err_px: Math.round(Math.sqrt(best.f) * 480) };
  localStorage.setItem(calKey(), JSON.stringify(calib));
  syncSliders();
  alert(`Calibrated ${camDef().label}. Reprojection error ≈ ${calib.err_px}px. FOV ${calib.fov.toFixed(1)}°`);
}

// ---------------- manual view adjustment sliders ----------------
const manualPanel = document.getElementById('manualcal');
function ensureCalib() {
  if (!calib) calib = { dp: [0, 0, 0], eul: [0, 0, 0], fov: +fovSlider.value };
  if (!calib.eul) {
    const q = calib.dq ? new THREE.Quaternion().fromArray(calib.dq) : new THREE.Quaternion();
    const e = new THREE.Euler().setFromQuaternion(q);
    calib.eul = [e.x, e.y, e.z];
    delete calib.dq;
  }
  return calib;
}
function syncSliders() {
  const c = calib || { dp: [0, 0, 0], eul: [0, 0, 0], fov: +fovSlider.value };
  const eul = c.eul || [0, 0, 0];
  document.querySelectorAll('.mc').forEach((s) => {
    const k = s.dataset.k;
    if (k === 'dx') s.value = c.dp[0];
    if (k === 'dy') s.value = c.dp[1];
    if (k === 'dz') s.value = c.dp[2];
    if (k === 'rx') s.value = THREE.MathUtils.radToDeg(eul[0]);
    if (k === 'ry') s.value = THREE.MathUtils.radToDeg(eul[1]);
    if (k === 'rz') s.value = THREE.MathUtils.radToDeg(eul[2]);
    if (k === 'fov') s.value = c.fov;
  });
}
document.querySelectorAll('.mc').forEach((s) => s.oninput = () => {
  const c = ensureCalib();
  const k = s.dataset.k, v = +s.value;
  if (k === 'dx') c.dp[0] = v;
  if (k === 'dy') c.dp[1] = v;
  if (k === 'dz') c.dp[2] = v;
  if (k === 'rx') c.eul[0] = THREE.MathUtils.degToRad(v);
  if (k === 'ry') c.eul[1] = THREE.MathUtils.degToRad(v);
  if (k === 'rz') c.eul[2] = THREE.MathUtils.degToRad(v);
  if (k === 'fov') c.fov = v;
  localStorage.setItem(calKey(), JSON.stringify(c));
  if (camMode) syncCamView();
});
document.getElementById('mcreset').onclick = () => {
  localStorage.removeItem(calKey());
  loadCalib();   // revert to the profile's default calib (authoritative FOV), else null
  syncSliders();
  if (camMode) syncCamView();
};

// ---------------- dream overlay (model's imagined future) ----------------
const dreamCanvas = document.getElementById('dreamframe');
const dreamCtl = document.getElementById('dreamctl');
const dreamBtn = document.getElementById('dreambtn');
let dreamOn = false, dreamSolo = false, dreamAlpha = 0.6, videoAlpha = 1.0;
const dreamImg = new Image();
// composite quadrants: TL head, TR wrist-right, BL wrist-left, BR unused
const DREAM_QUAD = { head: [0, 0], wr: [0.5, 0], wl: [0, 0.5] };
function dreamBase() { return (P && P.dreams && P.dreams[runName]) || null; }
function dreamFramePath() {
  return `${dreamBase()}/idx${replayData
    ? replayData.chunks[Math.min(Math.floor(t / replayData.EX), replayData.n - 1)].idx
    : samples[cur].idx}_t${replayData ? t % replayData.EX : t}.jpg`;
}
dreamImg.onload = () => {
  const ctx = dreamCanvas.getContext('2d');
  const q = DREAM_QUAD[camDef() ? camDef().id : 'head'] || [0, 0];
  const w = dreamImg.naturalWidth, h = dreamImg.naturalHeight;
  ctx.clearRect(0, 0, dreamCanvas.width, dreamCanvas.height);
  ctx.drawImage(dreamImg, q[0] * w, q[1] * h, w / 2, h / 2,
                0, 0, dreamCanvas.width, dreamCanvas.height);
};
function updateDream() {
  const avail = camMode && dreamBase();
  const active = avail && dreamOn;
  dreamBtn.style.display = avail ? '' : 'none';
  dreamCanvas.style.display = active ? 'block' : 'none';
  dreamCtl.style.display = active ? 'block' : 'none';
  arframe.style.opacity = active ? (dreamSolo ? 0 : videoAlpha) : 1;
  if (!active) return;
  dreamCanvas.style.opacity = dreamSolo ? 1 : dreamAlpha;
  dreamImg.src = dreamFramePath();
}
dreamBtn.onclick = () => {
  dreamOn = !dreamOn;
  dreamBtn.style.background = dreamOn ? '#3a4a6b' : '';
  updateDream();
};
document.getElementById('dreamsolo').onchange = (e) => { dreamSolo = e.target.checked; updateDream(); };
document.getElementById('dreamalpha').oninput = (e) => { dreamAlpha = e.target.value / 100; updateDream(); };
document.getElementById('videoalpha').oninput = (e) => { videoAlpha = e.target.value / 100; updateDream(); };

// ---------------- plan preview controls ----------------
document.getElementById('planshow').onchange = (e) => {
  planOn = e.target.checked;
  if (planBot) planBot.visible = planOn;
  applyPose();
};
document.getElementById('planahead').oninput = (e) => {
  planAhead = +e.target.value;
  document.getElementById('planaheadlabel').textContent = `+${planAhead}`;
  applyPose();
};

// ---------------- toggles / boot ----------------
document.getElementById('showpred').onchange = (e) => { if (predBot) predBot.visible = e.target.checked; };
document.getElementById('showgt').onchange = (e) => { if (gtBot) gtBot.visible = e.target.checked; };

await loadProfile(profileName);
(function loop() {
  applyCanvasRect();
  resize();
  if (camMode) syncCamView(); else controls.update();
  if (camMode && fisheyeOn) {
    // Pass 1: render the robot with a wide pinhole into the off-screen target.
    const fov0 = camera.fov, asp0 = camera.aspect;
    camera.fov = FISHEYE_RENDER_FOV; camera.aspect = IMG_ASPECT; camera.updateProjectionMatrix();
    renderer.setRenderTarget(fisheyeRT);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    camera.fov = fov0; camera.aspect = asp0; camera.updateProjectionMatrix();
    // Pass 2: resample through the fisheye model onto the canvas (over the video).
    const cd = camDef();
    const k = (cd && /wrist|wl|wr/i.test(cd.id)) ? FISHEYE_K.wrist : FISHEYE_K.head;
    fisheyeMat.uniforms.uK.value.set(k[0], k[1], k[2], k[3]);
    fisheyeMat.uniforms.uThetaEdge.value = THREE.MathUtils.degToRad(curFov()) / 2;
    renderer.render(fisheyeScene, fisheyeCam);
  } else {
    renderer.render(scene, camera);
  }
  requestAnimationFrame(loop);
})();
