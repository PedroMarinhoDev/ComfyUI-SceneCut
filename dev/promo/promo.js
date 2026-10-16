// The promo video, as a page rendered one frame at a time: renderFrame(t) puts every element where it is at
// t seconds. The viewer and the export dialog are the pack's own code (web/), driven through their methods; a
// cursor is drawn on top. Timers and performance.now() run on the video's clock, not the wall clock, so toasts,
// polling and progress keep their real durations however long each frame takes to capture.

// ---- virtual clock (before the pack's modules are loaded) -------------------------------------------
let T = 0;
const timers = [];
let timerSeq = 1;
window.setTimeout = (fn, ms = 0, ...args) => {
  const id = timerSeq++;
  timers.push({ id, due: T * 1000 + (+ms || 0), fn: () => fn(...args) });
  return id;
};
window.clearTimeout = (id) => { const i = timers.findIndex((x) => x.id === id); if (i >= 0) timers.splice(i, 1); };
Object.defineProperty(performance, "now", { value: () => T * 1000, configurable: true });
function runTimers() {
  for (let guard = 0; guard < 1000; guard++) {
    timers.sort((a, b) => a.due - b.due || a.id - b.id);
    const next = timers[0];
    if (!next || next.due > T * 1000 + 1e-6) return;
    timers.shift();
    try { next.fn(); } catch (e) { console.error(e); }
  }
}

const { SceneCutViewer, scrollWithin } = await import("./viewer_core.js");
const { openExportDialog, scopesFor } = await import("./export_dialog.js");

// ---- helpers ---------------------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const seg = (t, a, b) => clamp((t - a) / (b - a));
const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const easeOut = (x) => 1 - Math.pow(1 - x, 3);
const lerp = (a, b, k) => a + (b - a) * k;
const FPS = 24;
const pad = (n) => String(n).padStart(4, "0");
const BIG = (n) => `frames/${pad(n)}.jpg`;
const SMALL = (n) => `small/${pad(n)}.jpg`;
const pending = [];
function setImg(img, src) {
  if (img.dataset.src === src) return;
  img.dataset.src = src;
  img.src = src;
  pending.push(img.decode().catch(() => {}));
}
const raf = () => new Promise((r) => requestAnimationFrame(() => r()));
const XF = 0.4;                                   // crossfade between scenes
const vis = (t, a, b, first = false, last = false) =>
  Math.min(first ? 1 : seg(t, a - XF / 2, a + XF / 2), last ? 1 : 1 - seg(t, b - XF / 2, b + XF / 2));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---- data --------------------------------------------------------------------------------------------
const data = await (await fetch("data.json")).json();
const segs = data.segments;
const NF = data.source.frames;
const SELECT = [1, 6, 9, 14];
const FIX_TRUE = 492, FIX_WRONG = 494;           // checked: auto_fix brings 494 back to 492 on this video
delete data.selection;                          // the site's demo data comes with a selection; start with none
const variant = JSON.parse(JSON.stringify(data));
variant.base = "promo-fix";
variant.segments[5].end = FIX_WRONG;
variant.segments[6].start = FIX_WRONG;
delete variant.selection;

// ---- timeline --------------------------------------------------------------------------------------------
const S = { title: [0, 4], detect: [4, 14], review: [14, 24], fix: [24, 36], export: [36, 43.2], files: [43.2, 46],
            workflow: [46, 51], features: [51, 56], outro: [56, 64] };
const DURATION = 64;
const sfx = [];
const events = [];
const at = (t, fn, sound) => { events.push({ t, fn, done: false }); if (sound) sfx.push({ t, kind: sound }); };
for (const t of [4, 14, 24, 36, 43.2, 46, 51, 56]) sfx.push({ t: t - 0.15, kind: "whoosh" });

// ---- mosaics (title and outro) ------------------------------------------------------------------------
function buildMosaic(el) {
  const rows = [];
  for (let r = 0; r < 4; r++) {
    const row = document.createElement("div");
    row.className = "mrow";
    const imgs = [];
    for (let c = 0; c < 6; c++) {
      const img = document.createElement("img");
      row.append(img);
      imgs.push({ img, seg: segs[1 + ((r * 6 + c * 5) % 15)], phase: r * 7 + c * 13 });
    }
    el.append(row);
    rows.push({ row, imgs, dir: r % 2 ? 1 : -1 });
  }
  return rows;
}
const mosaics = [buildMosaic($("mosaic1")), buildMosaic($("mosaic2"))];
function updateMosaic(rows, t) {
  for (const { row, imgs, dir } of rows) {
    row.style.transform = `translateX(${-335 + dir * (t * 26 - 60)}px)`;
    for (const m of imgs) {
      const len = m.seg.end - m.seg.start;
      setImg(m.img, SMALL(m.seg.start + Math.floor(t * FPS + m.phase) % len));
    }
  }
}

// ---- 01 detection ----------------------------------------------------------------------------------------
const D0 = 4.35, D1 = 13.75;                     // the scan runs over the whole video in this window
const scan = $("scan"), sg = scan.getContext("2d");
const thumbs = segs.map((s) => { const i = new Image(); i.src = SMALL(s.start + Math.floor((s.end - s.start) / 2)); return i; });
await Promise.all(thumbs.map((i) => i.decode().catch(() => {})));
const revealT = (f) => D0 + (f / (NF - 1)) * (D1 - D0);
const cutKind = (i) => (segs[i].relation === "transition" ? "transition" : segs[i].relation === "sudden_jump" ? "jump" : "cut");
const chipText = (i) => {
  if (cutKind(i) === "jump") return "jump cut";
  if (cutKind(i) === "transition") return (segs[i - 1].absorbed || []).slice(-1)[0]?.transition || "transition";
  return null;
};
for (let i = 1; i < segs.length; i++) sfx.push({ t: revealT(segs[i].start), kind: chipText(i) ? "blip2" : "blip" });
function drawScan(t) {
  const W = 1728, H = 150, top = 52, bh = 92;
  const fpos = seg(t, D0, D1) * (NF - 1);
  const X = (f) => (f / NF) * W;
  sg.setTransform(2, 0, 0, 2, 0, 0);
  sg.clearRect(0, 0, W, H);
  sg.fillStyle = "#161b22";
  sg.beginPath(); sg.roundRect(0, top, W, bh, 10); sg.fill();
  sg.save();
  sg.beginPath(); sg.roundRect(0, top, W, bh, 10); sg.clip();
  for (let i = 0; i < segs.length; i++) {
    const s = segs[i];
    if (s.start > fpos) break;
    const x0 = X(s.start), x1 = X(s.end), xr = X(Math.min(s.end, fpos));
    sg.save();
    sg.beginPath(); sg.rect(x0, top, Math.max(0, xr - x0), bh); sg.clip();
    sg.fillStyle = i % 2 ? "#262b33" : "#1e232b";
    sg.fillRect(x0, top, x1 - x0, bh);
    const th = thumbs[i], tw = bh * 16 / 9;
    if (th.complete) {
      if (x1 - x0 >= tw) sg.drawImage(th, x0 + (x1 - x0 - tw) / 2, top, tw, bh);
      else { const sw = th.naturalWidth * (x1 - x0) / tw; sg.drawImage(th, (th.naturalWidth - sw) / 2, 0, sw, th.naturalHeight, x0, top, x1 - x0, bh); }
    }
    for (const a of s.absorbed || []) {
      sg.fillStyle = css("--trans");
      sg.fillRect(X(a.start), top + bh - 9, X(a.end) - X(a.start), 9);
    }
    sg.restore();
  }
  sg.restore();
  const placed = [];                                 // [x0, x1, row] of chips drawn so far
  for (let i = 1; i < segs.length; i++) {
    const s = segs[i];
    if (s.start > fpos) break;
    const k = easeOut(seg(t, revealT(s.start), revealT(s.start) + 0.22));
    const color = css(cutKind(i) === "jump" ? "--jump" : cutKind(i) === "transition" ? "--trans" : "--cut");
    const x = X(s.start);
    sg.fillStyle = color;
    const h = bh * k;
    sg.fillRect(x - 2, top + (bh - h) / 2, 4, h);
    const label = chipText(i);
    if (label) {
      sg.globalAlpha = k;
      sg.font = "600 20px Oswald";
      const w = sg.measureText(label.toUpperCase()).width + 18;
      let cx = clamp(x, w / 2, W - w / 2);
      const clash = placed.find(([a, b]) => cx - w / 2 < b + 6 && cx + w / 2 > a - 6);
      if (clash) cx = Math.max(cx, clash[1] + 8 + w / 2);    // slide right of the chip it would cover
      placed.push([cx - w / 2, cx + w / 2]);
      sg.fillStyle = color;
      sg.beginPath(); sg.roundRect(cx - w / 2, 8 + (1 - k) * 10, w, 32, 7); sg.fill();
      if (Math.abs(cx - x) > w / 2) {                        // a thin leader down to its cut
        sg.fillRect(x - 1, 40 + (1 - k) * 10, 2, top - 40);
      }
      sg.fillStyle = "#0D1117";
      sg.textAlign = "center"; sg.textBaseline = "middle";
      sg.fillText(label.toUpperCase(), cx, 25 + (1 - k) * 10);
      sg.globalAlpha = 1;
    }
  }
  const px = X(fpos + 0.5);
  sg.fillStyle = "#fff";
  sg.fillRect(px - 1.5, top - 8, 3, bh + 16);
  // counter
  let shots = 0, trans = 0, jumps = 0;
  for (let i = 0; i < segs.length; i++) {
    if (segs[i].start > fpos) break;
    shots++;
    if (i > 0 && cutKind(i) === "jump") jumps++;
    for (const a of segs[i].absorbed || []) if (a.start <= fpos) trans++;
  }
  const pl = (n, one, many) => `<b>${n}</b> ${n === 1 ? one : many}`;
  $("scan-count").innerHTML = `${pl(shots, "shot", "shots")} · ${pl(trans, "transition", "transitions")} · ${pl(jumps, "jump cut", "jump cuts")}`;
  setImg($("detect-img"), BIG(Math.round(fpos)));
}

// ---- the viewer (02-04) --------------------------------------------------------------------------------------
let fixResolve = null;
const viewer = new SceneCutViewer($("host"), {
  resolve: (n) => n,
  onReviewChange: () => {},
  onFix: () => new Promise((res) => { fixResolve = res; setTimeout(() => res({ cut: FIX_WRONG, to: FIX_TRUE, mode: "auto" }), 600); }),
  onExport: ({ selected, segs: vs, source }) => openExportDialog({
    fetchJSON: fakeFetch, source, scopes: scopesFor(vs, selected), scope: "selected",
    request: (scope) => ({ data: "promo", scope }),
  }),
});
const still = document.createElement("img");
still.className = "still";
function prepStill() {
  if (viewer.video) {
    viewer.video.removeAttribute("src");
    viewer.video.style.display = "none";
  }
  if (!still.isConnected || still.parentElement !== viewer.videoBox) viewer.videoBox.prepend(still);
}
viewer.load(data);
prepStill();
let vframe = -1;
function viewerFrame(n) {
  n = clamp(Math.round(n), 0, NF - 1);
  if (n !== vframe) { viewer._setFrame(n, false); vframe = n; }
  setImg(still, BIG(n));
}

// camera over the stage
const IDENT = { tx: 0, ty: 0, s: 1 };
const camAt = (fx, fy, s, qx, qy) => ({ tx: qx - fx * s, ty: qy - fy * s, s });
const HOST = { x: 130, y: 6, s: 1.2 };              // #host in promo.html
const hostToStage = (xh, yh) => [HOST.x + HOST.s * xh, HOST.y + HOST.s * yh];
const FIXCAM = camAt(...hostToStage(600, 200), 1.6, 960, 262);
function camBlend(a, b, k) { return { tx: lerp(a.tx, b.tx, k), ty: lerp(a.ty, b.ty, k), s: lerp(a.s, b.s, k) }; }
function camera(t) {
  if (t < S.fix[0]) { const s = lerp(1, 1.035, seg(t, 14, 24)); return camAt(960, 425, s, 960, 425); }
  const push = camAt(960, 425, 1.035, 960, 425);
  if (t < 26) return camBlend(push, FIXCAM, ease(seg(t, 24.25, 25.5)));
  if (t < 35) return camBlend(FIXCAM, IDENT, ease(seg(t, 33.0, 34.3)));
  return IDENT;
}

// ---- cursor ---------------------------------------------------------------------------------------------
const cursorEl = $("cursor"), rippleEl = $("ripple"), keyEl = $("keycap");
const center = (el, dx = 0, dy = 0) => {
  let last = null;                                  // where it was, once the element is gone (a closed dialog)
  return () => {
    const e = el();
    if (e && e.isConnected) { const r = e.getBoundingClientRect(); last = [r.left + r.width / 2 + dx, r.top + r.height / 2 + dy]; }
    return last || [1990, 1000];
  };
};
const fixed = (x, y) => () => [x, y];
const moves = [];       // {t, dur, to}
const clicks = [];      // times
const keys = [];        // [t0, t1]
const visible = [];     // [t0, t1]
const move = (t, dur, to) => moves.push({ t, dur, to });
const click = (t, fn) => { clicks.push(t); at(t, fn, "click"); };
function cursorAt(t) {
  let i = -1;
  for (let k = 0; k < moves.length; k++) if (moves[k].t <= t) i = k;
  if (i < 0) return moves[0]?.to() || [2000, 1200];
  const m = moves[i], to = m.to();
  const from = i > 0 ? moves[i - 1].to() : to;
  const k = ease(seg(t, m.t, m.t + m.dur));
  return [lerp(from[0], to[0], k), lerp(from[1], to[1], k)];
}
function updateCursor(t) {
  const on = visible.some(([a, b]) => t >= a && t <= b);
  cursorEl.style.display = rippleEl.style.display = keyEl.style.display = "none";
  if (!on) return;
  const [x, y] = cursorAt(t);
  cursorEl.style.display = "block";
  cursorEl.style.transform = `translate(${x - 7}px, ${y - 5}px)`;
  const c = clicks.filter((c) => t >= c && t < c + 0.45).pop();
  if (c != null) {
    const k = (t - c) / 0.45;
    rippleEl.style.display = "block";
    rippleEl.style.left = `${x}px`; rippleEl.style.top = `${y}px`;
    rippleEl.style.transform = `scale(${0.3 + 1.1 * easeOut(k)})`;
    rippleEl.style.opacity = String(1 - k);
  }
  if (keys.some(([a, b]) => t >= a && t <= b)) {
    keyEl.style.display = "block";
    keyEl.style.transform = `translate(${x + 26}px, ${y + 30}px)`;
  }
}

// ---- 02 review and select --------------------------------------------------------------------------------
const check = (i) => () => viewer.cards[i].querySelector(".scv-check");
const card = (i) => () => viewer.cards[i];
visible.push([15.1, 23.6]);
move(15.1, 0.01, fixed(1980, 1000));
move(15.5, 0.8, center(check(1)));
click(16.4, () => viewer.togglePick(1));
move(16.6, 0.7, center(check(6)));
click(17.4, () => viewer.togglePick(6));
move(17.6, 0.7, center(card(9), 0, -20));
keys.push([18.05, 18.7]);
click(18.4, () => { viewer.anchor = 9; viewer.togglePick(9); });
at(18.55, () => scrollWithin(viewer.strip, viewer.cards[14]));
move(18.7, 0.6, center(card(14), 0, -20));
keys.push([19.15, 19.8]);
click(19.45, () => { viewer.anchor = 14; viewer.togglePick(14); });
move(19.8, 0.7, center(() => viewer.onlyBtn));
click(20.6, () => viewer.toggleOnlyPicked());
move(20.9, 0.8, center(card(9), 0, -20));
click(21.9, () => {});
click(22.08, () => { viewer.focusScene(9); viewer.follow = true; });
move(22.6, 0.9, fixed(1990, 900));
function reviewFrame(t) {
  if (t < 22.08) return 72 + (t - 14) * FPS;
  return 690 + ((t - 22.08) * FPS) % 60;
}

// ---- 03 fix a cut --------------------------------------------------------------------------------------------
at(24.0, () => {
  viewer.load(variant);
  prepStill();
  vframe = -1;
  viewer.follow = false;
  viewer.onlyPicked = false;
  viewer._applyFilter();
  viewer.view = [425, 585];
  viewer._dirty = true;
});
const FS_FIRST = 488, FS_N = 10;
const fsFrames = $("fs-frames");
const fsCells = [];
for (let k = 0; k < FS_N; k++) {
  const f = document.createElement("div");
  f.className = "f";
  const img = document.createElement("img");
  const n = FS_FIRST + k;
  img.src = BIG(n);
  const lab = document.createElement("span");
  lab.textContent = String(n);
  f.append(img, lab);
  fsFrames.append(f);
  fsCells.push(f);
}
const fsMarker = document.createElement("div");
fsMarker.className = "marker";
const fsLabel = document.createElement("div");
fsLabel.className = "mlabel";
fsFrames.append(fsMarker, fsLabel);
const FIXED_AT = 30.25;                            // the stub answers 600 ms after the click at 29.65
function fixFrame(t) {
  if (t < 25.6) return FIX_TRUE - 4;
  if (t < 27.2) return FIX_TRUE - 4 + Math.floor((t - 25.6) / 0.4);
  return FIX_TRUE;
}
function updateFilmstrip(t) {
  const el = $("filmstrip");
  const k = easeOut(seg(t, 25.7, 26.2)) * (1 - seg(t, 32.6, 33.0));
  el.style.opacity = String(k);
  el.style.transform = `translateX(-50%) translateY(${(1 - k) * 20}px)`;
  const fixedK = ease(seg(t, FIXED_AT + 0.1, FIXED_AT + 0.6));
  const cellW = 168 + 8;
  const boundary = lerp(FIX_WRONG, FIX_TRUE, fixedK) - FS_FIRST;
  const x = boundary * cellW - 4;
  const done = t >= FIXED_AT + 0.1;
  const color = done ? css("--accent") : "#ff5d5d";
  fsMarker.style.left = `${x}px`;
  fsMarker.style.background = color;
  fsLabel.style.left = `${x}px`;
  fsLabel.style.background = color;
  fsLabel.style.color = "#0D1117";
  fsLabel.textContent = done ? `CUT FIXED · ${FIX_WRONG} → ${FIX_TRUE}` : "DETECTED CUT";
  fsCells.forEach((c, i) => c.classList.toggle("wrong", !done && FS_FIRST + i >= FIX_TRUE && FS_FIRST + i < FIX_WRONG));
  const note = $("fs-note");
  note.textContent = done ? "Moved 2 frames back, to where the picture changes" : `Frames ${FIX_TRUE}–${FIX_WRONG - 1} already show the next shot`;
  note.style.color = done ? css("--accent") : "#ff8a8a";
}
visible.push([27.6, 31.9]);
move(27.6, 0.01, fixed(1990, 1000));
move(28.0, 1.2, center(() => viewer.fixBtn));
click(29.65, () => viewer.fix("auto"));
sfx.push({ t: FIXED_AT, kind: "success" });
move(30.9, 0.9, fixed(1990, 1040));

// ---- 04 export ------------------------------------------------------------------------------------------------
at(35.8, () => {
  viewer.follow = true;
  viewer.fit();
  viewer.picked = new Set(SELECT);
  viewer.onlyPicked = false;
  viewer._applyFilter();
  viewer.focus = -1;
  viewer._markCards();
  viewer._updateCounts();
  vframe = -1;
});
const EXPORT_START = 39.15, EXPORT_LEN = 2.55;
const TOTAL = SELECT.reduce((n, i) => n + segs[i].end - segs[i].start, 0);
const STEM = "scenecut-demo";
const FILES = SELECT.map((i) => `${STEM}_${String(i + 1).padStart(3, "0")}.mp4`);
const FOLDER = "ComfyUI/output/scenecut/scenecut-demo/scenes_00001";
const CODECS = [
  { id: "h264", label: "H.264 (mp4)", gpu: false, quality: true, ext: ".mp4" },
  { id: "h264_nvenc", label: "H.264, NVIDIA GPU (mp4)", gpu: true, quality: true, ext: ".mp4" },
  { id: "h265", label: "H.265 / HEVC (mp4)", gpu: false, quality: true, ext: ".mp4" },
  { id: "h265_10bit", label: "H.265 10-bit (mp4)", gpu: false, quality: true, ext: ".mp4" },
  { id: "hevc_nvenc", label: "H.265, NVIDIA GPU (mp4)", gpu: true, quality: true, ext: ".mp4" },
  { id: "prores", label: "ProRes 422 HQ (mov)", gpu: false, quality: false, ext: ".mov" },
  { id: "ffv1", label: "FFV1 lossless (mkv)", gpu: false, quality: false, ext: ".mkv" },
];
function exportStatus() {
  const frac = seg(T, EXPORT_START, EXPORT_START + EXPORT_LEN);
  const done = frac >= 1;
  return { id: "promo", state: done ? "done" : "running", done: Math.round(frac * TOTAL), total: TOTAL,
           clips: Math.min(4, Math.floor(frac * 4 + 1e-6)), clipsTotal: 4, folder: FOLDER,
           relative: "output/scenecut/scenecut-demo/scenes_00001", codec: "h264", files: done ? FILES : undefined };
}
async function fakeFetch(path, body) {
  if (path === "/scenecut/export/codecs") return { codecs: CODECS, default: "h264", quality: ["high", "medium", "small"], defaultQuality: "high", jsonOnly: "none" };
  return exportStatus();
}
visible.push([36.3, 43.1]);
move(36.3, 0.01, fixed(1990, 980));
move(36.5, 0.9, center(() => viewer.exportBtn));
click(37.6, () => viewer.exportScenes());
move(38.1, 0.8, center(() => document.querySelector(".scx button.scx-go")));
click(39.1, () => document.querySelector(".scx button.scx-go").click());
sfx.push({ t: 41.95, kind: "success" });
move(42.1, 0.7, center(() => document.querySelector(".scx footer button.scx-go")));
click(42.95, () => document.querySelector(".scx footer button.scx-go")?.click());
function exportFrame(t) { return 72 + Math.max(0, t - 36) * FPS; }
function updateDialogBar(t) {
  const fill = document.querySelector(".scx .scx-bar > i");
  if (fill && t >= EXPORT_START) fill.style.width = `${Math.round(seg(t, EXPORT_START, EXPORT_START + EXPORT_LEN) * 100)}%`;
}

// ---- files ---------------------------------------------------------------------------------------------------
const fileImgs = SELECT.map((i, k) => {
  const s = segs[i];
  const el = document.createElement("div");
  el.className = "file";
  const img = document.createElement("img");
  const len = (s.end - s.start) / data.source.fps;
  el.innerHTML = `<div class="meta"><div class="name">${FILES[k]}</div><div class="info">${len.toFixed(2)} s · ${s.end - s.start} frames · H.264 + AAC</div></div>`;
  el.prepend(img);
  $("files").append(el);
  return { el, img, s };
});
$("jsonbox").innerHTML = `<b>scenes.json</b>  <i>// the scene list, your fixes, and:</i>
"clips": [
` + SELECT.map((i, k) => `  { "scene": ${i + 1}, "file": "${FILES[k]}", "start": ${segs[i].start}, "end": ${segs[i].end} }`).join(",\n") + "\n]";
function updateFiles(t) {
  fileImgs.forEach(({ el, img, s }, k) => {
    const a = easeOut(seg(t, 43.35 + k * 0.14, 43.75 + k * 0.14));
    el.style.opacity = String(a);
    el.style.transform = `translateY(${(1 - a) * 30}px)`;
    const len = s.end - s.start;
    setImg(img, SMALL(s.start + Math.floor((t - 43.2) * FPS) % len));
  });
  const j = easeOut(seg(t, 43.9, 44.4));
  $("jsonbox").style.opacity = String(j);
}

// ---- workflow and features --------------------------------------------------------------------------------------
const FEATS = [
  ["validate", "Checks every cut", "DINOv3 merges flashes, strobes and whip pans back into their shot."],
  ["detect", "Presets", "Film, music video, anime, vlog and sports. In seconds, not frames."],
  ["select", "Selected Scenes", "Play your selection scene by scene, in the node or its own tab."],
  ["frames", "Images in and out", "Scenes as IMAGE batches for any workflow; frames in as a scene."],
  ["review", "Frame-exact", "Frame n is the n-th decoded frame, in every stage and every clip."],
  ["export", "Runs anywhere", "GPU or CPU, no ffmpeg needed, and a core that runs without ComfyUI."],
];
const featEls = FEATS.map(([icon, title, text]) => {
  const el = document.createElement("div");
  el.className = "feat";
  el.innerHTML = `<img src="icons/${icon}.svg" alt=""><h3>${title}</h3><p>${text}</p>`;
  $("grid").append(el);
  return el;
});

// ---- outro ----------------------------------------------------------------------------------------------------
const outroParts = [...document.querySelectorAll("#L-outro .titlebox > *")];

// ---- frame update -----------------------------------------------------------------------------------------------
function update(t) {
  const L = (id, a, b, first, last) => { const v = vis(t, a, b, first, last); $(id).style.opacity = String(v); return v > 0; };
  if (L("L-title", ...S.title, true)) {
    updateMosaic(mosaics[0], t);
    const k = easeOut(seg(t, 0, 3.8));
    document.querySelector("#L-title .titlebox").style.transform = `scale(${lerp(1.03, 1, k)})`;
  }
  if (L("L-detect", ...S.detect)) drawScan(t);
  const dip = 1 - Math.sin(Math.PI * seg(t, 23.85, 24.15)) * 0.9;
  const stageOn = vis(t, S.review[0], S.export[1]) * dip;
  $("L-stage").style.opacity = String(stageOn);
  if (stageOn > 0 || (t > S.review[0] - 0.3 && t < S.files[0] + 0.3)) {
    const c = camera(t);
    $("stage").style.transform = `translate(${c.tx}px, ${c.ty}px) scale(${c.s})`;
    if (t < S.fix[0]) viewerFrame(reviewFrame(t));
    else if (t < 35.8) viewerFrame(fixFrame(t));
    else viewerFrame(exportFrame(t));
  }
  L("L-review", ...S.review);
  if (L("L-fix", ...S.fix)) updateFilmstrip(t);
  L("L-export", ...S.export);
  if (L("L-files", ...S.files)) updateFiles(t);
  if (L("L-workflow", ...S.workflow)) {
    const k = seg(t, S.workflow[0] - 0.2, S.workflow[1] + 0.2);
    $("shot").style.transform = `translateX(-50%) scale(${lerp(1, 1.07, k)})`;
  }
  if (L("L-features", ...S.features)) {
    featEls.forEach((el, i) => {
      const a = easeOut(seg(t, 51.15 + i * 0.16, 51.6 + i * 0.16));
      el.style.opacity = String(a);
      el.style.transform = `translateY(${(1 - a) * 40}px) scale(${lerp(0.96, 1, a)})`;
    });
  }
  if (L("L-outro", ...S.outro, false, true)) {
    updateMosaic(mosaics[1], t);
    outroParts.forEach((el, i) => {
      const a = easeOut(seg(t, 56.1 + i * 0.22, 56.6 + i * 0.22));
      el.style.opacity = String(a);
      el.style.transform = `translateY(${(1 - a) * 24}px)`;
    });
  }
  updateDialogBar(t);
  updateCursor(t);
}

// ---- driver -------------------------------------------------------------------------------------------------------
events.sort((a, b) => a.t - b.t);
const fontsReady = Promise.all([
  document.fonts.load("400 90px Anton"), document.fonts.load("600 40px Oswald"),
  document.fonts.load("400 30px 'IBM Plex Sans'"), document.fonts.load("600 30px 'IBM Plex Sans'"),
  document.fonts.load("500 24px 'IBM Plex Mono'"),
]);
await fontsReady;
window.renderFrame = async (t) => {
  T = t;
  for (const e of events) if (!e.done && e.t <= t) { e.done = true; try { e.fn(); } catch (err) { console.error(err); } }
  runTimers();
  await Promise.resolve();
  runTimers();
  update(t);
  await Promise.all(pending.splice(0));
  await raf();
  update(t);                                        // positions that depend on layout done in this frame
  await Promise.all(pending.splice(0));
  await raf();
  await raf();
  return t;
};
window.PROMO = { duration: DURATION, sfx: sfx.sort((a, b) => a.t - b.t) };
window.promoReady = true;
