// SceneCut export dialog: clips (one file per scene, frame-exact, with sound) and scenes.json,
// written by the server to output/scenecut/<video>/scenes_NNNNN/. Shared by the SceneCut Viewer,
// the Selected Scenes node and the Selected Scenes page.
//
//   openExportDialog({
//     fetchJSON(path, body?) -> Promise<object>,   // "/scenecut/..." routes; POST when body is given
//     source: { name, fps, hasAudio },
//     scopes: [{ id, label, count, frames, disabled? }],  // what can be exported
//     scope,                                       // the scope selected first
//     request(scopeId) -> { data, review, scope, indices? },
//     onFinished(status),                          // also called when the dialog was hidden
//   })

const CSS = `
.scx-back { position: fixed; inset: 0; z-index: 10050; background: rgba(0,0,0,.55); display: flex;
  align-items: center; justify-content: center; font: 13px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }
.scx { width: min(520px, 94vw); max-height: 92vh; overflow: auto; background: #1c1c21; color: #e6e6ea;
  border: 1px solid #3a3a43; border-radius: 12px; box-shadow: 0 16px 50px rgba(0,0,0,.55); }
.scx header { display: flex; align-items: center; gap: 10px; padding: 14px 16px 10px; }
.scx header h3 { margin: 0; font-size: 15px; flex: 1; }
.scx header .scx-x { background: none; border: 0; color: #aaa; font-size: 20px; cursor: pointer; line-height: 1; }
.scx .scx-src { padding: 0 16px 10px; color: #9a9aa6; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.scx .scx-body { padding: 4px 16px 6px; display: grid; gap: 14px; }
.scx .scx-row > label.scx-h { display: block; color: #9a9aa6; font-size: 12px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: .4px; }
.scx .scx-opts { display: grid; gap: 6px; }
.scx .scx-opt { display: flex; align-items: center; gap: 10px; padding: 8px 10px; border: 1px solid #34343c;
  border-radius: 8px; cursor: pointer; }
.scx .scx-opt:hover { border-color: #55555f; }
.scx .scx-opt.scx-on { border-color: #3ecf6e; background: #1c2a21; }
.scx .scx-opt.scx-off { opacity: .45; cursor: default; }
.scx .scx-opt b { font-weight: 600; }
.scx .scx-opt span { color: #9a9aa6; margin-left: auto; font-variant-numeric: tabular-nums; }
.scx select { width: 100%; background: #2a2a31; color: #e6e6ea; border: 1px solid #3a3a43; border-radius: 6px;
  padding: 6px 8px; font: inherit; }
.scx .scx-two { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.scx .scx-check { display: flex; align-items: center; gap: 8px; color: #ccc; }
.scx .scx-note { color: #9a9aa6; font-size: 12px; }
.scx footer { display: flex; align-items: center; gap: 8px; padding: 12px 16px 16px; }
.scx footer .scx-sp { flex: 1; }
.scx button.scx-b { background: #2a2a31; color: #e6e6ea; border: 1px solid #3a3a43; border-radius: 7px;
  padding: 7px 14px; font: inherit; cursor: pointer; }
.scx button.scx-b:hover:not(:disabled) { background: #34343c; }
.scx button.scx-b:disabled { opacity: .45; cursor: default; }
.scx button.scx-go { background: #2e8b57; border-color: #2e8b57; color: #fff; font-weight: 600; }
.scx button.scx-go:hover:not(:disabled) { background: #36a066; }
.scx .scx-bar { height: 8px; border-radius: 4px; background: #2a2a31; overflow: hidden; margin: 6px 0 8px; }
.scx .scx-bar > i { display: block; height: 100%; width: 0; background: #3ecf6e; transition: width .3s; }
.scx .scx-stat { display: flex; justify-content: space-between; color: #9a9aa6; font-variant-numeric: tabular-nums; }
.scx .scx-path { margin-top: 8px; padding: 8px 10px; background: #141418; border: 1px solid #2c2c33; border-radius: 6px;
  font: 12px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace; word-break: break-all; user-select: text; }
.scx .scx-ok { color: #6fdc8c; font-weight: 600; }
.scx .scx-err { color: #ff8a80; }
`;

const PREFS = "SceneCut.export.prefs";
let openDialog = null;          // the dialog on screen (one at a time)

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function ensureStyle() {
  if (document.getElementById("scx-style")) return;
  const s = el("style");
  s.id = "scx-style";
  s.textContent = CSS;
  document.head.append(s);
}

function fmtDur(sec) {
  if (!isFinite(sec) || sec < 0) return "";
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = Math.floor(sec % 60);
  return h ? `${h} h ${m} min` : m ? `${m} min ${s} s` : `${s} s`;
}

function loadPrefs() {
  try { return JSON.parse(localStorage.getItem(PREFS) || "{}"); } catch { return {}; }
}

function savePrefs(p) {
  try { localStorage.setItem(PREFS, JSON.stringify(p)); } catch { /* private mode */ }
}

export function openExportDialog(opts) {
  ensureStyle();
  openDialog?.close();
  const prefs = loadPrefs();
  const fps = opts.source?.fps || 25;
  let scope = opts.scope || opts.scopes.find((s) => !s.disabled)?.id;
  let job = null, timer = null, hidden = false, running = false, t0 = 0;

  const back = el("div", "scx-back");
  const box = el("div", "scx");
  back.append(box);
  const head = el("header");
  const title = el("h3", null, "Export scenes");
  const x = el("button", "scx-x", "×");
  x.title = "Close (Esc)";
  head.append(title, x);
  const src = el("div", "scx-src", opts.source?.name || "");
  const body = el("div", "scx-body");
  const foot = el("footer");
  box.append(head, src, body, foot);

  // ---- form
  const scopeRow = el("div", "scx-row");
  scopeRow.append(el("label", "scx-h", "Scenes"));
  const scopeOpts = el("div", "scx-opts");
  scopeRow.append(scopeOpts);
  const renderScopes = () => {
    scopeOpts.replaceChildren(...opts.scopes.map((s) => {
      const o = el("div", "scx-opt" + (s.id === scope ? " scx-on" : "") + (s.disabled ? " scx-off" : ""));
      o.append(el("b", null, s.label), el("span", null,
        `${s.count} scene${s.count === 1 ? "" : "s"} · ${fmtDur(s.frames / fps) || "0 s"}`));
      if (!s.disabled) o.addEventListener("click", () => { scope = s.id; renderScopes(); update(); });
      return o;
    }));
  };
  const codecRow = el("div", "scx-row");
  codecRow.append(el("label", "scx-h", "Clips"));
  const codecSel = el("select");
  codecSel.append(new Option("Loading...", ""));
  const two = el("div", "scx-two");
  const qualSel = el("select");
  for (const [v, t] of [["high", "High quality"], ["medium", "Medium"], ["small", "Small files"]]) qualSel.append(new Option(t, v));
  qualSel.value = prefs.quality || "high";
  const audioLbl = el("label", "scx-check");
  const audioBox = el("input");
  audioBox.type = "checkbox";
  audioBox.checked = prefs.audio !== false && !!opts.source?.hasAudio;
  audioBox.disabled = !opts.source?.hasAudio;
  audioLbl.append(audioBox, opts.source?.hasAudio ? "Include sound" : "No sound in this video");
  two.append(qualSel, audioLbl);
  const note = el("div", "scx-note", "");
  codecRow.append(codecSel, two, note);
  two.style.marginTop = "8px";
  note.style.marginTop = "8px";
  body.append(scopeRow, codecRow);

  const cancelBtn = el("button", "scx-b", "Cancel");
  const goBtn = el("button", "scx-b scx-go", "Export");
  foot.append(el("span", "scx-sp"), cancelBtn, goBtn);

  let codecs = [], jsonOnly = "none";
  const update = () => {
    const c = codecs.find((k) => k.id === codecSel.value);
    const json = codecSel.value === jsonOnly;
    qualSel.disabled = json || (c && !c.quality);
    audioBox.disabled = json || !opts.source?.hasAudio;
    const s = opts.scopes.find((k) => k.id === scope);
    goBtn.disabled = !s || s.disabled || !codecSel.value;
    goBtn.textContent = json ? "Save scene list" : `Export ${s?.count ?? 0} clip${s?.count === 1 ? "" : "s"}`;
    note.textContent = json
      ? "Writes scenes.json only (all scenes, with the selection and your cut fixes)."
      : `One ${c?.ext || ".mp4"} file per scene, cut on the exact frames, plus scenes.json. `
        + (c?.gpu ? "Encodes on the NVIDIA GPU." : c?.id?.startsWith("pro") || c?.id === "ffv1" ? "Large files, for editing." : "Encodes on the CPU.");
  };
  codecSel.addEventListener("change", update);

  opts.fetchJSON("/scenecut/export/codecs").then((r) => {
    codecs = r.codecs || [];
    jsonOnly = r.jsonOnly || "none";
    codecSel.replaceChildren(...codecs.map((c) => new Option(c.label, c.id)),
      new Option("No clips: scene list only", jsonOnly));
    const want = prefs.codec && [...codecSel.options].some((o) => o.value === prefs.codec) ? prefs.codec : r.default;
    codecSel.value = want;
    update();
  }).catch((e) => { note.textContent = `Could not list codecs: ${e.message || e}`; note.className = "scx-note scx-err"; });

  // ---- running / finished
  const showProgress = (st) => {
    title.textContent = st.state === "done" ? "Export finished" : st.state === "running" ? "Exporting..." :
      st.state === "cancelled" ? "Export cancelled" : "Export failed";
    body.replaceChildren();
    if (st.state === "running" || st.state === "done") {
      const frac = st.total ? st.done / st.total : st.state === "done" ? 1 : 0;
      const bar = el("div", "scx-bar");
      const fill = el("i");
      fill.style.width = `${Math.round(frac * 100)}%`;
      bar.append(fill);
      const stat = el("div", "scx-stat");
      const elapsed = (performance.now() - t0) / 1000;
      const eta = st.state === "running" && frac > 0.02 ? elapsed * (1 - frac) / frac : NaN;
      stat.append(el("span", null, st.clipsTotal ? `clip ${Math.min(st.clips + (st.state === "running" ? 1 : 0), st.clipsTotal)} of ${st.clipsTotal}` : "scene list"),
        el("span", null, st.state === "running" ? `${Math.round(frac * 100)}%${isFinite(eta) ? ` · about ${fmtDur(eta)} left` : ""}` : `took ${fmtDur(elapsed) || "1 s"}`));
      body.append(bar, stat);
    }
    if (st.state === "done") {
      const n = st.files?.length || 0;
      body.prepend(el("div", "scx-ok", n ? `Saved ${n} clip${n === 1 ? "" : "s"} and scenes.json` : "Saved scenes.json"));
    }
    if (st.state === "error") body.append(el("div", "scx-err", st.error || "Unknown error"));
    const path = el("div", "scx-path", st.folder || "");
    path.title = st.relative || "";
    if (st.folder && st.state !== "cancelled") body.append(path);
    foot.replaceChildren(el("span", "scx-sp"));
    if (st.state === "running") {
      const stop = el("button", "scx-b", "Cancel export");
      stop.addEventListener("click", () => opts.fetchJSON(`/scenecut/export/${st.id}/cancel`, {}).catch(() => {}));
      const hide = el("button", "scx-b", "Keep running in background");
      hide.addEventListener("click", close);
      foot.append(stop, hide);
    } else {
      if (st.folder && st.state === "done" && navigator.clipboard) {
        const copy = el("button", "scx-b", "Copy folder path");
        copy.addEventListener("click", () => navigator.clipboard.writeText(st.folder).then(() => { copy.textContent = "Copied"; }));
        foot.append(copy);
      }
      const ok = el("button", "scx-b scx-go", "Close");
      ok.addEventListener("click", close);
      foot.append(ok);
    }
  };

  const poll = async () => {
    try {
      const st = await opts.fetchJSON(`/scenecut/export/${job}`);
      if (!hidden) showProgress(st);
      if (st.state === "running") { timer = setTimeout(poll, 600); return; }
      running = false;
      opts.onFinished?.(st, hidden);
    } catch (e) {
      if (!hidden) showProgress({ state: "error", error: e.message || String(e) });
    }
  };

  goBtn.addEventListener("click", async () => {
    savePrefs({ codec: codecSel.value, quality: qualSel.value, audio: audioBox.checked });
    goBtn.disabled = true;
    try {
      const st = await opts.fetchJSON("/scenecut/export", {
        ...opts.request(scope), codec: codecSel.value, quality: qualSel.value, audio: audioBox.checked,
      });
      if (st.error) throw new Error(st.error);
      job = st.id;
      running = true;
      t0 = performance.now();
      showProgress(st);
      timer = setTimeout(poll, 400);
    } catch (e) {
      goBtn.disabled = false;
      note.textContent = e.message || String(e);
      note.className = "scx-note scx-err";
    }
  });

  function close() {
    back.remove();
    document.removeEventListener("keydown", onKey, true);
    if (openDialog === handle) openDialog = null;
    if (running) hidden = true;                   // the export goes on; a toast reports the end
    else clearTimeout(timer);
  }
  const onKey = (e) => {
    if (e.key === "Escape") { e.preventDefault(); close(); }
    e.stopPropagation();                          // keep the canvas and the viewer from reacting
  };
  x.addEventListener("click", close);
  cancelBtn.addEventListener("click", close);
  back.addEventListener("mousedown", (e) => { if (e.target === back) close(); });
  document.addEventListener("keydown", onKey, true);
  renderScopes();
  update();
  document.body.append(back);
  const handle = { close };
  openDialog = handle;
  return handle;
}

/** Scopes for a set of segments: the selected ones and all of them. */
export function scopesFor(segs, selected) {
  const frames = (idx) => idx.reduce((n, i) => n + (segs[i].end - segs[i].start), 0);
  const sel = [...selected].filter((i) => i >= 0 && i < segs.length);
  return [
    { id: "selected", label: "Selected scenes", count: sel.length, frames: frames(sel), disabled: !sel.length },
    { id: "all", label: "All scenes", count: segs.length, frames: frames(segs.map((_, i) => i)) },
  ];
}
