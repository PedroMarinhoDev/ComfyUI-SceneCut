"""assets/workflow.png: the basic workflow in ComfyUI after a run on the demo video.

    python dev/comfy_screenshot.py [--server http://127.0.0.1:8188]

Drives headless Chrome over the DevTools protocol: opens the SceneCut-basic template on a running
ComfyUI (with SceneCut installed), runs it on dev/demo/scenecut-demo.mp4, selects four scenes and
takes a 2x screenshot. Headless Chrome doesn't paint video frames, so the players show the frame
they are on as a still image. Run it against a server with its own --user-directory: it changes
settings. Needs ComfyUI's Python (aiohttp) and google-chrome.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scenecut.media import iter_frames_at  # noqa: E402

DEMO = ROOT / "dev" / "demo" / "scenecut-demo.mp4"
W, H = 1720, 1060


def still(frame: int, width: int = 960) -> str:
    from PIL import Image

    rgb = dict(iter_frames_at(str(DEMO), [frame], width, width * 9 // 16))[frame]
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


SCRIPT = r"""
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  for (let i = 0; i < 100 && !(window.comfyAPI?.app?.app?.graph?.nodes?.length); i++) await sleep(200);
  const app = window.comfyAPI.app.app, { api } = window.comfyAPI.api;
  await app.extensionManager.setting.set("Comfy.VueNodes.Enabled", false);
  await app.extensionManager.setting.set("Comfy.Minimap.Visible", false);
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
  const lv = app.graph.nodes.find((n) => n.type === "SceneCutLoadVideo");
  const w = lv.widgets.find((w) => w.name === "path");
  w.value = VIDEO; w.callback(w.value);
  // validation is calibrated for camera footage; the generated demo runs without it, as on the page
  const det = app.graph.nodes.find((n) => n.type === "SceneCutDetect");
  det.widgets.find((x) => x.name === "validate").value = false;
  await sleep(1200);
  // fit the four nodes (DOM widgets only mount on screen) between the side bar and the top bar
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const n of app.graph.nodes) {
    x0 = Math.min(x0, n.pos[0]); y0 = Math.min(y0, n.pos[1] - 30);
    x1 = Math.max(x1, n.pos[0] + n.size[0]); y1 = Math.max(y1, n.pos[1] + n.size[1]);
  }
  const left = 84, top = 64, right = 20, bottom = 20;
  const scale = Math.min((innerWidth - left - right) / (x1 - x0), (innerHeight - top - bottom) / (y1 - y0));
  app.canvas.ds.scale = scale;
  app.canvas.ds.offset = [left / scale - x0, top / scale - y0];
  app.canvas.setDirty(true, true);
  let done = false;
  api.addEventListener("execution_success", () => { done = true; }, { once: true });
  await app.queuePrompt(0, 1);
  for (let i = 0; i < 600 && !done; i++) await sleep(250);
  for (let i = 0; i < 120 && !document.querySelector(".scv-root")?.sceneCutViewer?.data; i++) await sleep(250);
  for (let i = 0; i < 120 && !document.querySelector(".scs-root")?.sceneCutSelected?.data; i++) await sleep(250);
  const v = document.querySelector(".scv-root").sceneCutViewer;
  v.setReview({ base: v.data.base, edits: [], selection: [1, 6, 9, 13] });
  v.opts.onReviewChange?.(v.review());
  v.focusScene(9);
  v._setFrame(v.segs[9].start + 20, false);
  await sleep(800);
  const p = document.querySelector(".scs-root").sceneCutSelected;
  p.go(1, false);
  await sleep(500);
  const swap = (video, src) => {
    if (!video) return;
    const img = document.createElement("img");
    img.src = src;
    img.style.cssText = getComputedStyle(video).cssText;
    img.style.objectFit = "contain"; img.style.display = "block";
    img.style.maxWidth = "100%"; img.style.maxHeight = "100%"; img.style.height = "100%"; img.style.width = "auto";
    video.style.display = "none";
    video.parentElement.insertBefore(img, video);
  };
  swap(v.video, STILL_VIEWER);
  swap(p.video, STILL_SELECTED);
  const preview = document.querySelector("video[src*='scenecut/media']");
  if (preview) { preview.poster = STILL_PREVIEW; preview.removeAttribute("src"); preview.load(); }
  w.value = "~/Videos/scenecut-demo.mp4";       // shown instead of this machine's path (no callback: the preview stays)
  document.querySelectorAll(".p-toast-message").forEach((t) => t.remove());
  app.canvas.setDirty(true, true);
  await sleep(1500);
  return JSON.stringify({ scenes: v.segs.length, selected: [...v.picked] });
})()
"""


async def main(server: str, out: Path) -> None:
    profile = tempfile.mkdtemp()
    chrome = subprocess.Popen(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars", "--remote-debugging-port=9333",
                               f"--user-data-dir={profile}", f"--window-size={W},{H}", "--force-device-scale-factor=2", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with aiohttp.ClientSession() as http:
            for _ in range(50):
                try:
                    pages = await (await http.get("http://127.0.0.1:9333/json")).json()
                    break
                except aiohttp.ClientError:
                    await asyncio.sleep(0.2)
            target = next(p for p in pages if p["type"] == "page")
            async with http.ws_connect(target["webSocketDebuggerUrl"], max_msg_size=0) as ws:
                ids = iter(range(1, 10_000))

                async def call(method, **params):
                    i = next(ids)
                    await ws.send_json({"id": i, "method": method, "params": params})
                    while True:
                        msg = json.loads((await ws.receive()).data)
                        if msg.get("id") == i:
                            if "error" in msg:
                                raise RuntimeError(msg["error"])
                            return msg["result"]

                await call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=2, mobile=False)
                await call("Page.navigate", url=f"{server}/?template=SceneCut-basic&source=ComfyUI-SceneCut")
                await asyncio.sleep(6)
                script = (SCRIPT.replace("VIDEO", json.dumps(str(DEMO)))
                          .replace("STILL_VIEWER", json.dumps(still(710)))
                          .replace("STILL_SELECTED", json.dumps(still(492)))
                          .replace("STILL_PREVIEW", json.dumps(still(120)))
                          )
                res = await call("Runtime.evaluate", expression=script, awaitPromise=True, timeout=180000)
                print(res.get("result", {}).get("value") or res.get("exceptionDetails", {}).get("exception", {}).get("description", res))
                shot = await call("Page.captureScreenshot", format="png")
                out.write_bytes(base64.b64decode(shot["data"]))
                print(f"wrote {out}")
    finally:
        chrome.terminate()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--out", default=str(ROOT / "assets" / "workflow.png"))
    a = p.parse_args()
    asyncio.run(main(a.server, Path(a.out)))
