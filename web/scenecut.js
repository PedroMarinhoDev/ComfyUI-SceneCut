// SceneCut frontend:
//   - a server-side file picker and a playable preview for "SceneCut Load Video (Path)"
//   - the "SceneCut Viewer" node (timeline, cut fixes, selection, save buttons)
//   - the "SceneCut Selected Scenes" node (follows the viewer's selection; opens in its own tab)
//
// The browser's own file dialog never reveals a file's real path, so picking a video with it
// would mean uploading (copying) the whole file into ComfyUI. The picker lists the folders of
// the machine ComfyUI runs on instead and fills in the path; nothing is copied.

import { SceneCutViewer } from "./viewer_core.js";
import { SceneCutSelected, channelName, openSelectedExport } from "./selected_core.js";
import { openExportDialog, scopesFor } from "./export_dialog.js";

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const LAST_DIR_KEY = "SceneCut.browse.lastDir";
const STYLE_ID = "scenecut-browse-style";

const CSS = `
.sc-overlay { position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,.55);
  display: flex; align-items: center; justify-content: center; }
.sc-dialog { width: min(820px, 94vw); height: min(600px, 88vh); display: flex; flex-direction: column;
  background: var(--comfy-menu-bg, #202020); color: var(--fg-color, #ddd);
  border: 1px solid var(--border-color, #444); border-radius: 10px; overflow: hidden;
  box-shadow: 0 12px 40px rgba(0,0,0,.5); font: 13px/1.4 system-ui, sans-serif; }
.sc-head { display: flex; align-items: center; gap: 8px; padding: 10px 12px;
  border-bottom: 1px solid var(--border-color, #444); }
.sc-head h3 { margin: 0; font-size: 14px; font-weight: 600; flex: 1; }
.sc-bar { display: flex; gap: 6px; padding: 8px 12px; border-bottom: 1px solid var(--border-color, #444); }
.sc-bar input { flex: 1; min-width: 0; }
.sc-dialog input[type=text] { background: var(--comfy-input-bg, #2a2a2a); color: var(--input-text, #eee);
  border: 1px solid var(--border-color, #444); border-radius: 6px; padding: 5px 8px; font: inherit; }
.sc-dialog button { background: var(--comfy-input-bg, #2a2a2a); color: var(--input-text, #eee);
  border: 1px solid var(--border-color, #444); border-radius: 6px; padding: 5px 12px; font: inherit;
  cursor: pointer; }
.sc-dialog button:hover:not(:disabled) { filter: brightness(1.25); }
.sc-dialog button:disabled { opacity: .45; cursor: default; }
.sc-dialog button.sc-primary { background: #2f6fd6; border-color: #2f6fd6; color: #fff; }
.sc-main { flex: 1; display: flex; min-height: 0; }
.sc-places { width: 150px; flex: none; overflow-y: auto; padding: 6px;
  border-right: 1px solid var(--border-color, #444); }
.sc-place { padding: 5px 8px; border-radius: 6px; cursor: pointer; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; color: var(--descrip-text, #aaa); }
.sc-place:hover { background: rgba(255,255,255,.07); color: var(--fg-color, #ddd); }
.sc-list { flex: 1; overflow-y: auto; padding: 4px 0; }
.sc-row { display: grid; grid-template-columns: 22px 1fr 84px 92px; align-items: center; gap: 6px;
  padding: 4px 12px; cursor: pointer; user-select: none; }
.sc-row:hover { background: rgba(255,255,255,.06); }
.sc-row.sc-sel { background: rgba(47,111,214,.35); }
.sc-row .sc-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sc-row .sc-meta { color: var(--descrip-text, #999); font-size: 12px; text-align: right;
  font-variant-numeric: tabular-nums; }
.sc-icon { width: 16px; height: 16px; opacity: .8; }
.sc-msg { padding: 16px; color: var(--descrip-text, #999); }
.sc-msg.sc-err { color: #e57373; }
.sc-foot { display: flex; align-items: center; gap: 10px; padding: 10px 12px;
  border-top: 1px solid var(--border-color, #444); }
.sc-foot label { display: flex; align-items: center; gap: 4px; color: var(--descrip-text, #aaa); }
.sc-foot .sc-chosen { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; color: var(--descrip-text, #aaa); text-align: right; }
.sc-note { padding: 6px 12px; font-size: 12px; color: #e0b050;
  border-bottom: 1px solid var(--border-color, #444); }
`;

const ICONS = {
  folder: '<svg class="sc-icon" viewBox="0 0 24 24" fill="#d8a83c"><path d="M3 6.5A1.5 1.5 0 0 1 4.5 5h5l2 2.2h8A1.5 1.5 0 0 1 21 8.7v9.8a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 18.5z"/></svg>',
  video: '<svg class="sc-icon" viewBox="0 0 24 24" fill="#6fa8ff"><path d="M4 5h11a2 2 0 0 1 2 2v2.4l4-2.4v10l-4-2.4V17a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z"/></svg>',
  file: '<svg class="sc-icon" viewBox="0 0 24 24" fill="#999"><path d="M6 3h8l4 4v14H6z"/></svg>',
};

function h(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v);
  }
  for (const c of children) if (c != null) e.append(c);
  return e;
}

function fmtSize(n) {
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n < 10 && i ? n.toFixed(1) : Math.round(n)} ${u[i]}`;
}

function fmtDate(t) {
  const d = new Date(t * 1000);
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function joinPath(dir, name, sep) {
  return dir.endsWith(sep) ? dir + name : dir + sep + name;
}

async function fetchListing(path, opts) {
  const q = new URLSearchParams({ path: path || "", all: opts.all ? "1" : "0", hidden: opts.hidden ? "1" : "0" });
  const r = await api.fetchApi(`/scenecut/browse?${q}`);
  const data = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok) {
    const err = new Error(data.error || `HTTP ${r.status}`);
    err.data = data;
    throw err;
  }
  return data;
}

function ensureStyle() {
  if (!document.getElementById(STYLE_ID)) {
    document.head.append(h("style", { id: STYLE_ID }, CSS));
  }
}

/** Open the picker. Resolves to the chosen absolute path, or null when cancelled. */
function pickVideo(start) {
  ensureStyle();
  return new Promise((resolve) => {
    const opts = { all: false, hidden: false };
    let current = null;      // last listing
    let selected = null;     // file name selected in `current`

    const pathInput = h("input", { type: "text", spellcheck: "false", placeholder: "Folder path" });
    const filterInput = h("input", { type: "text", spellcheck: "false", placeholder: "Filter", style: "width:140px" });
    const upBtn = h("button", { title: "Parent folder (Backspace)" }, "Up");
    const list = h("div", { class: "sc-list", tabindex: "0" });
    const places = h("div", { class: "sc-places" });
    const note = h("div", { class: "sc-note", style: "display:none" });
    const chosen = h("div", { class: "sc-chosen" });
    const selectBtn = h("button", { class: "sc-primary", disabled: "" }, "Select");
    const allBox = h("input", { type: "checkbox" });
    const hiddenBox = h("input", { type: "checkbox" });

    const close = (value) => {
      overlay.remove();
      document.removeEventListener("keydown", onKey, true);
      resolve(value);
    };
    const confirm = () => {
      if (!current || !selected) return;
      localStorage.setItem(LAST_DIR_KEY, current.path);
      close(joinPath(current.path, selected, current.sep));
    };

    const setSelected = (name) => {
      selected = name;
      for (const row of list.querySelectorAll(".sc-row")) {
        row.classList.toggle("sc-sel", row.dataset.file === name);
      }
      chosen.textContent = name || "";
      if (name) selectBtn.removeAttribute("disabled"); else selectBtn.setAttribute("disabled", "");
    };

    const renderPlaces = (items) => {
      places.replaceChildren(...(items || []).map((p) =>
        h("div", { class: "sc-place", title: p.path, onclick: () => go(p.path) }, p.name)));
    };

    const render = () => {
      const f = filterInput.value.trim().toLowerCase();
      const keep = (n) => !f || n.toLowerCase().includes(f);
      const rows = [];
      for (const d of current.dirs.filter(keep)) {
        rows.push(h("div", { class: "sc-row", title: d, onclick: () => go(joinPath(current.path, d, current.sep)) },
          h("span", { html: ICONS.folder }), h("span", { class: "sc-name" }, d), h("span"), h("span")));
      }
      for (const file of current.files.filter((x) => keep(x.name))) {
        const row = h("div", {
          class: "sc-row", title: file.name,
          onclick: () => setSelected(file.name),
          ondblclick: () => { setSelected(file.name); confirm(); },
        },
          h("span", { html: file.video ? ICONS.video : ICONS.file }),
          h("span", { class: "sc-name" }, file.name),
          h("span", { class: "sc-meta" }, fmtSize(file.size)),
          h("span", { class: "sc-meta" }, fmtDate(file.mtime)));
        row.dataset.file = file.name;
        rows.push(row);
      }
      if (!rows.length) {
        rows.push(h("div", { class: "sc-msg" }, f ? "Nothing matches the filter." :
          (opts.all ? "This folder is empty." : "No folders or videos here.")));
      }
      if (current.truncated) rows.push(h("div", { class: "sc-msg" }, "Showing the first 5000 entries; use the filter."));
      list.replaceChildren(...rows);
      setSelected(current.files.some((x) => x.name === selected) ? selected : null);
    };

    const go = async (path, pick) => {
      list.replaceChildren(h("div", { class: "sc-msg" }, "Loading..."));
      try {
        current = await fetchListing(path, opts);
      } catch (e) {
        list.replaceChildren(h("div", { class: "sc-msg sc-err" }, e.message));
        if (e.data?.places) renderPlaces(e.data.places);
        return;
      }
      pathInput.value = current.path;
      filterInput.value = "";
      upBtn.disabled = !current.parent;
      renderPlaces(current.places);
      note.style.display = current.restricted ? "" : "none";
      note.textContent = current.restricted
        ? "Browsing is limited to the folders listed on the left (set SCENECUT_BROWSE_ROOTS on the server to allow others)."
        : "";
      selected = pick || null;
      render();
      list.focus();
    };

    const onKey = (e) => {
      const typing = e.target === pathInput || e.target === filterInput;
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(null); }
      else if (e.key === "Enter" && e.target === pathInput) { e.preventDefault(); go(pathInput.value); }
      else if (e.key === "Enter" && !typing) { e.preventDefault(); confirm(); }
      else if (e.key === "Backspace" && !typing && current?.parent) { e.preventDefault(); go(current.parent); }
      else return;
      e.stopPropagation();
    };

    upBtn.addEventListener("click", () => current?.parent && go(current.parent));
    filterInput.addEventListener("input", () => current && render());
    selectBtn.addEventListener("click", confirm);
    allBox.addEventListener("change", () => { opts.all = allBox.checked; current && go(current.path, selected); });
    hiddenBox.addEventListener("change", () => { opts.hidden = hiddenBox.checked; current && go(current.path, selected); });

    const overlay = h("div", { class: "sc-overlay", onmousedown: (e) => { if (e.target === overlay) close(null); } },
      h("div", { class: "sc-dialog", role: "dialog", "aria-label": "Select a video" },
        h("div", { class: "sc-head" },
          h("h3", {}, "Select a video"),
          h("button", { title: "Close (Esc)", onclick: () => close(null) }, "×")),
        h("div", { class: "sc-bar" }, upBtn, pathInput, filterInput),
        note,
        h("div", { class: "sc-main" }, places, list),
        h("div", { class: "sc-foot" },
          h("label", {}, allBox, "All files"),
          h("label", {}, hiddenBox, "Hidden"),
          chosen,
          h("button", { onclick: () => close(null) }, "Cancel"),
          selectBtn)));

    document.addEventListener("keydown", onKey, true);
    document.body.append(overlay);

    // Start in the folder of the current value, else where the last pick was made.
    const startFile = start && /\.[a-z0-9]{2,5}$/i.test(start) ? start.split(/[\\/]/).pop() : null;
    go(start || localStorage.getItem(LAST_DIR_KEY) || "", startFile);
  });
}

app.registerExtension({
  name: "SceneCut.LoadVideoBrowse",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SceneCutLoadVideo") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const pathWidget = this.widgets?.find((w) => w.name === "path");
      const button = this.addWidget("button", "Browse...", null, async () => {
        const value = (pathWidget?.value || "").trim().replace(/^["']|["']$/g, "");
        const picked = await pickVideo(value);
        if (picked && pathWidget) {
          pathWidget.value = picked;
          pathWidget.callback?.(picked);
          this.setDirtyCanvas?.(true, true);
          app.graph.setDirtyCanvas(true, true);
        }
      }, { serialize: false });
      button.serialize = false;
      addVideoPreview(this, pathWidget);
      return r;
    };
  },
});



// ------------------------------------------------------------------------------------------------
// Shared helpers

const bus = (window.__sceneCutBus ||= new EventTarget());

function viewUrl(name, subfolder = "scenecut", type = "temp") {
  const q = new URLSearchParams({ filename: name, subfolder, type });
  return api.apiURL(`/view?${q}`);
}

async function postJSON(route, body) {
  const r = await api.fetchApi(route, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const d = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok && !d.error) d.error = `HTTP ${r.status}`;
  return d;
}

function hideWidget(w) {
  if (!w) return;
  w.hidden = true;
  w.computeSize = () => [0, -4];
}

function cleanPath(v) {
  return (v || "").trim().replace(/^["']|["']$/g, "");
}

function fmtDuration(sec) {
  const m = Math.floor(sec / 60), s = sec - m * 60;
  return m ? `${m}:${s.toFixed(1).padStart(4, "0")}` : `${s.toFixed(1)} s`;
}

const NAV_HINT = "Hold <kbd>Ctrl</kbd> + scroll to zoom the canvas · <kbd>Ctrl</kbd> + drag to move it";

/**
 * Canvas navigation over a node's DOM widget: Ctrl+wheel zooms the canvas, Ctrl+drag or
 * middle-drag pans it. A plain wheel scrolls whatever can scroll under the pointer; elsewhere it
 * shows a hint, so nobody gets stuck zoomed in on a big widget. `exclude` (the timeline) keeps its
 * own wheel handling. Returns a cleanup function.
 */
function bindCanvasNav(host, { exclude = null, inOverlay = () => false, hint = () => {} } = {}) {
  // Nodes 2.0 forwards every wheel event over a node to the canvas before it reaches the widget,
  // unless the target is inside a focused [data-capture-wheel] element (Ctrl+wheel still goes to
  // the canvas). Clicking the widget focuses it, so the wheel works inside it after a click.
  host.dataset.captureWheel = "true";
  if (host.tabIndex < 0 && !host.hasAttribute("tabindex")) host.tabIndex = -1;
  const canScroll = (el, e) => {
    const cs = getComputedStyle(el);
    const dy = e.deltaY;
    if (/(auto|scroll)/.test(cs.overflowY) && el.scrollHeight > el.clientHeight + 1) {
      if ((dy > 0 && el.scrollTop + el.clientHeight < el.scrollHeight - 1) || (dy < 0 && el.scrollTop > 0)) return "y";
    }
    if (/(auto|scroll)/.test(cs.overflowX) && el.scrollWidth > el.clientWidth + 1) {
      const d = Math.abs(e.deltaX) > Math.abs(dy) ? e.deltaX : dy;
      if ((d > 0 && el.scrollLeft + el.clientWidth < el.scrollWidth - 1) || (d < 0 && el.scrollLeft > 0)) return "x";
    }
    return null;
  };
  const onWheel = (e) => {
    if (inOverlay()) return;
    if (exclude && e.target.closest?.(exclude)) return;
    e.stopPropagation();
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      app.canvas?.processMouseWheel?.(e);
      return;
    }
    for (let el = e.target; el && el !== host.parentElement; el = el.parentElement) {
      const dir = canScroll(el, e);
      if (dir === "y") return;                          // native vertical scroll
      if (dir === "x") {                                // a horizontal strip: scroll it with the wheel
        e.preventDefault();
        el.scrollLeft += Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        return;
      }
      if (el === host) break;
    }
    e.preventDefault();
    hint();
  };
  let pan = null;
  const onDown = (e) => {
    if (inOverlay()) return;
    const middle = e.button === 1, ctrl = e.button === 0 && (e.ctrlKey || e.metaKey);
    if (!middle && !ctrl) return;
    if (ctrl && exclude && e.target.closest?.(exclude)) return;   // ctrl+click there has its own meaning
    pan = { x: e.clientX, y: e.clientY, moved: false };
    if (middle) { e.preventDefault(); e.stopPropagation(); }
  };
  const onMove = (e) => {
    if (!pan) return;
    if (!pan.moved && Math.hypot(e.clientX - pan.x, e.clientY - pan.y) < 4) return;
    pan.moved = true;
    const ds = app.canvas?.ds;
    if (ds) {
      ds.offset[0] += (e.clientX - pan.x) / ds.scale;
      ds.offset[1] += (e.clientY - pan.y) / ds.scale;
      app.canvas.setDirty(true, true);
    }
    pan.x = e.clientX; pan.y = e.clientY;
    e.preventDefault();
  };
  const onUp = () => {
    if (pan?.moved) {
      // swallow the click that ends a drag (it would toggle a card's selection)
      const stop = (ev) => { ev.stopPropagation(); ev.preventDefault(); };
      window.addEventListener("click", stop, { capture: true, once: true });
      setTimeout(() => window.removeEventListener("click", stop, { capture: true }), 0);
    }
    pan = null;
  };
  host.addEventListener("wheel", onWheel, { passive: false, capture: true });
  host.addEventListener("pointerdown", onDown, { capture: true });
  window.addEventListener("pointermove", onMove, true);
  window.addEventListener("pointerup", onUp, true);
  return () => {
    host.removeEventListener("wheel", onWheel, { capture: true });
    host.removeEventListener("pointerdown", onDown, { capture: true });
    window.removeEventListener("pointermove", onMove, true);
    window.removeEventListener("pointerup", onUp, true);
  };
}

/** Full-window mode for a DOM widget's content. */
function makeOverlay(getRoot, host, onChange) {
  let overlay = null, onEsc = null;
  const close = () => {
    if (!overlay) return;
    host.append(getRoot());
    overlay.remove();
    overlay = null;
    document.removeEventListener("keydown", onEsc, true);
    onChange(false);
  };
  const open = () => {
    if (overlay) return close();
    overlay = document.createElement("div");
    overlay.className = "scv-overlay";
    overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
    overlay.append(getRoot());
    document.body.append(overlay);
    onEsc = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
    document.addEventListener("keydown", onEsc, true);
    getRoot().focus({ preventScroll: true });
    onChange(true);
  };
  return { open, close, isOpen: () => !!overlay };
}

function makeStatus(host, text) {
  const status = document.createElement("div");
  status.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;"
    + "color:#888;font:12px system-ui;background:#16161a;border-radius:6px;padding:12px;text-align:center;";
  status.textContent = text;
  host.append(status);
  return (t) => { status.textContent = t; status.style.display = t ? "flex" : "none"; };
}

/** Store the workflow now: node.properties changes go unnoticed by the workflow's change tracker
 * until something else changes, so a reload right after a run would lose the viewer's results. */
function persistGraph() {
  try { app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.(); } catch { /* older frontends */ }
}

function toastInfo(detail, severity = "info", life = 5000) {
  app.extensionManager?.toast?.add?.({ severity, summary: "SceneCut", detail, life });
}

async function fetchJSON(path, body) {
  if (body === undefined) {
    const r = await api.fetchApi(path);
    const d = await r.json().catch(() => ({ error: r.statusText }));
    if (!r.ok && !d.error) d.error = `HTTP ${r.status}`;
    return d;
  }
  return postJSON(path, body);
}

/** A toast when an export finishes (shown whether or not its dialog is still open). */
function exportFinished(st, hidden) {
  if (st.state === "done" && hidden) {
    const n = st.files?.length || 0;
    toastInfo(`Export finished: ${n ? `${n} clip${n === 1 ? "" : "s"} and ` : ""}scenes.json in ${st.folder}`, "success", 8000);
  } else if (st.state === "error") {
    toastInfo(`Export failed: ${st.error}`, "error", 10000);
  }
}

// ------------------------------------------------------------------------------------------------
// Load Video (Path): a small playable preview of the file

function addVideoPreview(node, pathWidget) {
  const box = document.createElement("div");
  box.style.cssText = "width:100%;height:100%;min-height:150px;display:flex;flex-direction:column;gap:4px;"
    + "background:#111;border-radius:6px;overflow:hidden;position:relative;";
  const video = document.createElement("video");
  video.controls = true;
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.style.cssText = "flex:1;min-height:0;width:100%;object-fit:contain;background:#000;display:none;";
  const info = document.createElement("div");
  info.style.cssText = "font:11px system-ui;color:#aaa;padding:2px 6px 4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
  const msg = document.createElement("div");
  msg.style.cssText = "flex:1;display:flex;align-items:center;justify-content:center;color:#777;font:12px system-ui;padding:8px;text-align:center;";
  msg.textContent = "Pick a video to preview it here.";
  box.append(video, msg, info);
  const hint = document.createElement("div");
  hint.className = "scv-hint";
  box.append(hint);
  const w = node.addDOMWidget("scenecut_preview", "div", box, { serialize: false, getMinHeight: () => 170 });
  w.serialize = false;
  bindCanvasNav(box, {
    hint: () => {
      hint.innerHTML = NAV_HINT;
      hint.classList.add("scv-show");
      clearTimeout(hint._t);
      hint._t = setTimeout(() => hint.classList.remove("scv-show"), 1800);
    },
  });

  let current = null, triedProxy = false, timer = null;
  const show = (text) => { msg.textContent = text; msg.style.display = text ? "flex" : "none"; video.style.display = text ? "none" : "block"; };
  const load = async (path) => {
    path = cleanPath(path);
    if (path === current) return;
    current = path;
    triedProxy = false;
    info.textContent = "";
    if (!path) { video.removeAttribute("src"); video.load(); show("Pick a video to preview it here."); return; }
    show("Loading...");
    const r = await api.fetchApi(`/scenecut/probe?${new URLSearchParams({ path })}`);
    const d = await r.json().catch(() => ({ error: r.statusText }));
    if (current !== path) return;
    if (!r.ok) { show(d.error || "Cannot preview this file."); return; }
    info.textContent = `${d.width}×${d.height} · ${d.fps_float.toFixed(3).replace(/\.?0+$/, "")} fps · ${fmtDuration(d.duration)} · ${d.codec}${d.has_audio ? " + audio" : ""}`;
    info.title = d.path;
    video.src = api.apiURL(`/scenecut/media?${new URLSearchParams({ path })}`);
    show("");
  };
  video.addEventListener("error", async () => {
    if (!current || triedProxy) { if (current) show("This video cannot be previewed in the browser."); return; }
    triedProxy = true;
    const path = current;
    show("The browser cannot play this file; making a small preview copy...");
    const r = await api.fetchApi(`/scenecut/preview?${new URLSearchParams({ path })}`);
    const d = await r.json().catch(() => ({ error: r.statusText }));
    if (current !== path) return;
    if (!r.ok) { show(d.error || "Cannot preview this file."); return; }
    video.src = viewUrl(d.filename, d.subfolder, d.type);
    show("");
  });
  if (pathWidget) {
    const cb = pathWidget.callback;
    pathWidget.callback = function (v) {
      const out = cb?.apply(this, arguments);
      clearTimeout(timer);
      timer = setTimeout(() => load(v), 400);
      return out;
    };
  }
  const onConfigure = node.onConfigure;
  node.onConfigure = function () {
    onConfigure?.apply(this, arguments);
    load(pathWidget?.value);
  };
  if (!node.size || node.size[1] < 300) node.setSize([Math.max(node.size?.[0] || 0, 360), 340]);
  setTimeout(() => load(pathWidget?.value), 0);
}

// ------------------------------------------------------------------------------------------------
// Connecting an images output turns its output_images option on.

app.registerExtension({
  name: "SceneCut.ImageOutputs",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!String(nodeData.name).startsWith("SceneCut")) return;
    if (!(nodeData.output || []).includes("IMAGE")) return;
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, slot, connected) {
      const r = onConnectionsChange?.apply(this, arguments);
      const OUTPUT = window.LiteGraph?.OUTPUT ?? 2;
      if (type !== OUTPUT || !connected || this.outputs?.[slot]?.type !== "IMAGE") return r;
      const w = this.widgets?.find((x) => x.name === "output_images");
      if (!w || w.value !== "off" || app.configuringGraph) return r;
      const viewerLike = nodeData.name === "SceneCutViewer" || nodeData.name === "SceneCutSelectedScenes";
      w.value = viewerLike ? "all frames" : "first frame per scene";
      w.callback?.(w.value);
      toastInfo(`${nodeData.display_name || nodeData.name}: output_images set to "${w.value}" (change it on the node).`);
      return r;
    };
  },
});

// ------------------------------------------------------------------------------------------------
// SceneCut Viewer node

app.registerExtension({
  name: "SceneCut.Viewer",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SceneCutViewer") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;
      const reviewW = this.widgets?.find((w) => w.name === "review");
      hideWidget(reviewW);
      const host = document.createElement("div");
      host.style.cssText = "width:100%;height:100%;min-height:420px;position:relative;";
      const setStatus = makeStatus(host, "Run the workflow to see the scenes here.");
      const widget = this.addDOMWidget("scenecut_viewer", "div", host, { serialize: false, getMinHeight: () => 420 });
      widget.serialize = false;          // display only: keep it out of widgets_values
      if (!this.size || this.size[0] < 760) this.setSize([840, 880]);

      let viewer = null, info = null;
      const overlay = makeOverlay(() => viewer.root, host, (on) => { viewer.setExpanded(on); viewer._layout(); });
      const unbindNav = bindCanvasNav(host, {
        exclude: ".scv-timeline", inOverlay: overlay.isOpen, hint: () => viewer?.hint(NAV_HINT),
      });

      const readReview = () => { try { return JSON.parse(reviewW?.value || "{}"); } catch { return {}; } };
      const publish = () => {
        node.sceneCutLive = viewer?.data ? { info, data: viewer.data, review: viewer.review() } : null;
        bus.dispatchEvent(new CustomEvent("viewer", { detail: { node } }));
      };
      const writeReview = (review) => {
        if (reviewW) {
          const empty = !review.edits.length && !review.selection.length;
          reviewW.value = empty ? "" : JSON.stringify(review);
          reviewW.callback?.(reviewW.value);
        }
        node.setDirtyCanvas?.(true, true);
        persistGraph();
        publish();
      };

      const load = async (newInfo) => {
        if (viewer?.data && info?.data === newInfo.data) { info = newInfo; return; }
        try {
          setStatus("Loading...");
          const res = await fetch(viewUrl(newInfo.data, newInfo.subfolder, newInfo.type));
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          info = newInfo;
          if (!viewer) {
            viewer = new SceneCutViewer(host, {
              resolve: (name) => viewUrl(name, info.subfolder, info.type),
              onExpand: overlay.open,
              onReviewChange: writeReview,
              onFix: (mode, frame, edits) => postJSON("/scenecut/review/fix", { data: info.data, mode, frame, edits }),
              onExport: ({ review, selected, segs, source }) => openExportDialog({
                fetchJSON, source,
                scopes: scopesFor(segs, selected),
                scope: selected.length ? "selected" : "all",
                request: (scope) => ({ data: info.data, review, scope }),
                onFinished: exportFinished,
              }),
            });
          }
          viewer.load(data);
          viewer.setReview(readReview());
          setStatus("");
          if (newInfo.report) viewer.toast(`HTML report saved: output/${newInfo.report}`, "ok", 6000);
          publish();
        } catch (err) {
          setStatus("Preview files are gone (ComfyUI empties its temp folder on restart). Run the node again.");
          console.warn("SceneCut viewer:", err);
        }
      };

      const onExecuted = this.onExecuted;
      this.onExecuted = function (output) {
        onExecuted?.apply(this, arguments);
        const entry = output?.scenecut_viewer?.[0];
        if (!entry) return;
        this.properties = this.properties || {};
        this.properties.scenecut_viewer = entry;
        persistGraph();
        load(entry);
      };
      const onConfigure = this.onConfigure;
      this.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        hideWidget(reviewW);
        const entry = this.properties?.scenecut_viewer;
        if (entry) load(entry);
      };
      const onRemoved = this.onRemoved;
      this.onRemoved = function () {
        onRemoved?.apply(this, arguments);
        overlay.close();
        unbindNav();
        viewer?.destroy();
        node.sceneCutLive = null;
        bus.dispatchEvent(new CustomEvent("viewer", { detail: { node } }));
      };
      return r;
    };
  },
});

// ------------------------------------------------------------------------------------------------
// SceneCut Selected Scenes node

function upstreamViewer(node) {
  const slot = node.inputs?.findIndex((i) => i.name === "scenes") ?? -1;
  const imagesSlot = node.inputs?.find((i) => i.name === "images");
  if (slot < 0 || imagesSlot?.link != null) return null;
  const up = node.getInputNode?.(slot);
  return up && (up.comfyClass === "SceneCutViewer" || up.type === "SceneCutViewer") ? up : null;
}

app.registerExtension({
  name: "SceneCut.SelectedScenes",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SceneCutSelectedScenes") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;
      this.properties = this.properties || {};
      this.properties.scenecut_channel ||= Math.random().toString(36).slice(2, 10);
      const host = document.createElement("div");
      host.style.cssText = "width:100%;height:100%;min-height:360px;position:relative;";
      const setStatus = makeStatus(host, "Connect a SceneCut Viewer (or a scene list) and run the workflow.");
      const widget = this.addDOMWidget("scenecut_selected", "div", host, { serialize: false, getMinHeight: () => 360 });
      widget.serialize = false;
      if (!this.size || this.size[0] < 560) this.setSize([660, 640]);

      let player = null, own = null, info = null, state = { edits: [], selection: null };
      let bc = null, bcName = null;
      const overlay = makeOverlay(() => player.root, host, (on) => player.setExpanded(on));
      const unbindNav = bindCanvasNav(host, {
        inOverlay: overlay.isOpen,
        hint: () => {
          if (!player) return;
          let h = player.root.querySelector(".scv-hint");
          if (!h) { h = document.createElement("div"); h.className = "scv-hint"; player.root.append(h); }
          h.innerHTML = NAV_HINT;
          h.classList.add("scv-show");
          clearTimeout(h._t);
          h._t = setTimeout(() => h.classList.remove("scv-show"), 1800);
        },
      });

      const broadcast = () => {
        if (!bc || !info) return;
        // plain copies: node properties are reactive proxies, which cannot be posted
        const msg = { type: "state", info, edits: state.edits, selection: state.selection, current: player?.current() };
        try { bc.postMessage(JSON.parse(JSON.stringify(msg))); } catch (err) { console.warn("SceneCut selected scenes:", err); }
      };
      const channel = () => {
        const name = channelName(node.properties.scenecut_channel);
        if (bc && bcName === name) return bc;
        bc?.close();
        bcName = name;
        bc = "BroadcastChannel" in window ? new BroadcastChannel(name) : null;
        if (bc) {
          bc.onmessage = (ev) => {
            const m = ev.data || {};
            if (m.type === "hello") broadcast();
            else if (m.type === "nav" && player) player.goScene(m.index);
          };
        }
        return bc;
      };
      const openTab = () => {
        if (!info) return;
        channel();
        const q = new URLSearchParams({
          ch: node.properties.scenecut_channel, data: info.data, subfolder: info.subfolder, type: info.type,
          sel: JSON.stringify(state.selection || null), edits: JSON.stringify(state.edits || []),
          title: player?.data?.source?.name || "Selected scenes",
        });
        window.open(api.apiURL(`/scenecut/selected?${q}`), `scenecut-${node.properties.scenecut_channel}`);
      };

      const refresh = async () => {
        const up = upstreamViewer(node);
        let src = null;
        if (up?.sceneCutLive) {
          const live = up.sceneCutLive;
          src = { info: live.info, edits: live.review.edits, selection: live.review.selection };
        } else if (own) {
          src = { info: own, edits: [], selection: own.selection };
        }
        if (!src) { setStatus("Connect a SceneCut Viewer (or a scene list) and run the workflow."); return; }
        state = { edits: [...(src.edits || [])].map((e) => ({ ...e })),
          selection: src.selection?.length ? [...src.selection] : null };
        try {
          if (!player) {
            player = new SceneCutSelected(host, {
              resolve: (name) => viewUrl(name, info?.subfolder, info?.type),
              onExpand: overlay.open,
              onOpenTab: openTab,
              onNavigate: (index) => channel()?.postMessage({ type: "nav", index }),
              onExport: (x) => openSelectedExport(x, () => info, () => state, fetchJSON, exportFinished),
            });
          }
          if (!player.data || info?.data !== src.info.data) {
            setStatus("Loading...");
            const res = await fetch(viewUrl(src.info.data, src.info.subfolder, src.info.type));
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            info = { data: src.info.data, subfolder: src.info.subfolder, type: src.info.type };
            player.load(await res.json(), state);
          } else {
            player.update(state);
          }
          setStatus("");
          channel();
          broadcast();
        } catch (err) {
          setStatus("Preview files are gone (ComfyUI empties its temp folder on restart). Run the workflow again.");
          console.warn("SceneCut selected scenes:", err);
        }
      };

      const onViewer = (e) => { if (e.detail.node === upstreamViewer(node)) refresh(); };
      bus.addEventListener("viewer", onViewer);

      const onConnectionsChange = this.onConnectionsChange;
      this.onConnectionsChange = function () {
        const out = onConnectionsChange?.apply(this, arguments);
        if (!app.configuringGraph) setTimeout(refresh, 0);
        return out;
      };
      const onExecuted = this.onExecuted;
      this.onExecuted = function (output) {
        onExecuted?.apply(this, arguments);
        const entry = output?.scenecut_selected?.[0];
        if (!entry) return;
        own = entry;
        this.properties.scenecut_selected = entry;
        persistGraph();
        refresh();
      };
      const onConfigure = this.onConfigure;
      this.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        own = this.properties?.scenecut_selected || null;
        setTimeout(refresh, 0);
      };
      const onRemoved = this.onRemoved;
      this.onRemoved = function () {
        onRemoved?.apply(this, arguments);
        bus.removeEventListener("viewer", onViewer);
        overlay.close();
        unbindNav();
        bc?.close();
        player?.destroy();
      };
      return r;
    };
  },
});
