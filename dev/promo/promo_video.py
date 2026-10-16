"""The promo video for X and Reddit: 1920x1080, 60 fps, 64 s, with music and sound effects.

    cp trailer.mp4 dev/demo/trailer.mp4 && python dev/site_assets.py   # the demo video and the viewer data
    python dev/promo/promo_video.py --stills 2,10,18,28,30.6,40,44,48,53,60   # check frames
    python dev/promo/promo_video.py                              # dev/promo/out/scenecut-promo.mp4

The picture is promo.html + promo.js rendered frame by frame by headless Chrome (the pack's own viewer
and export dialog, the demo trailer, a drawn cursor). The music is synthesized here: an A minor
progression at 120 BPM whose sections follow the video's, with clicks, blips, whooshes and chimes at
the times the page lists. Needs ComfyUI's Python (aiohttp, numpy, scipy), google-chrome and ffmpeg.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import http.server
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD, OUT = HERE / "build", HERE / "out"
DEMO = ROOT / "dev" / "demo" / "trailer.mp4"
FONTS = Path.home() / "Projects" / "music-visualizer" / "public" / "fonts"
PLEX = Path("/usr/share/fonts/truetype/ibm-plex")
FPS, W, H = 60, 1920, 1080
SR = 48000
PORT, CDP = 8194, 9335


# --------------------------------------------------------------------------------------------- #
# Page assets
# --------------------------------------------------------------------------------------------- #
def build() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    for name in ("promo.html", "promo.js"):
        shutil.copyfile(HERE / name, BUILD / name)
    for name in ("viewer_core.js", "export_dialog.js"):
        shutil.copyfile(ROOT / "web" / name, BUILD / name)
    site = ROOT / "site" / "demo"
    shutil.copyfile(site / "data.json", BUILD / "data.json")
    for f in site.glob("*.jpg"):
        shutil.copyfile(f, BUILD / f.name)
    (BUILD / "fonts").mkdir(exist_ok=True)
    shutil.copyfile(FONTS / "anton-latin-400-normal.woff2", BUILD / "fonts" / "anton.woff2")
    shutil.copyfile(FONTS / "oswald-latin-wght-normal.woff2", BUILD / "fonts" / "oswald.woff2")
    for f in ("IBMPlexSans-Regular", "IBMPlexSans-Medium", "IBMPlexSans-SemiBold", "IBMPlexSans-Bold",
              "IBMPlexMono-Regular", "IBMPlexMono-Medium"):
        shutil.copyfile(PLEX / f"{f}.ttf", BUILD / "fonts" / f"{f}.ttf")
    shutil.copytree(ROOT / "assets" / "icons", BUILD / "icons", dirs_exist_ok=True)
    shutil.copyfile(ROOT / "assets" / "workflow.png", BUILD / "workflow.png")
    for folder, size in (("frames", "960:540"), ("small", "480:270")):
        d = BUILD / folder
        if d.exists() and any(d.iterdir()):
            continue
        d.mkdir(exist_ok=True)
        # passthrough: one image per decoded frame (a file whose first frame starts late would otherwise
        # get a duplicate first image, and every number would be one frame off)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(DEMO), "-vf", f"scale={size}:flags=lanczos", "-q:v", "2",
                        "-fps_mode", "passthrough", "-start_number", "0", str(d / "%04d.jpg")], check=True)


def serve() -> http.server.ThreadingHTTPServer:
    handler = functools.partial(QuietHandler, directory=str(BUILD))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


# --------------------------------------------------------------------------------------------- #
# Rendering through the DevTools protocol
# --------------------------------------------------------------------------------------------- #
class Page:
    def __init__(self, ws):
        self.ws, self.seq = ws, 0

    async def call(self, method, **params):
        self.seq += 1
        await self.ws.send_json({"id": self.seq, "method": method, "params": params})
        while True:
            msg = json.loads((await self.ws.receive()).data)
            if msg.get("id") == self.seq:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg["result"]
            if msg.get("method") == "Runtime.exceptionThrown":
                print("page error:", msg["params"]["exceptionDetails"].get("exception", {}).get("description"), file=sys.stderr)
            if msg.get("method") == "Runtime.consoleAPICalled" and msg["params"]["type"] in ("error", "warning"):
                print("console:", [a.get("value", a.get("description")) for a in msg["params"]["args"]], file=sys.stderr)

    async def eval(self, expr, timeout=120000):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True, timeout=timeout)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r))
        return r["result"].get("value")

    async def shot(self) -> bytes:
        r = await self.call("Page.captureScreenshot", format="png",
                            clip={"x": 0, "y": 0, "width": W, "height": H, "scale": 0.5})
        return base64.b64decode(r["data"])


async def with_page(fn):
    import aiohttp

    profile = tempfile.mkdtemp()
    chrome = subprocess.Popen(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={CDP}",
                               f"--user-data-dir={profile}", f"--window-size={W},{H}", "--autoplay-policy=no-user-gesture-required",
                               "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with aiohttp.ClientSession() as http_:
            for _ in range(100):
                try:
                    pages = await (await http_.get(f"http://127.0.0.1:{CDP}/json")).json()
                    break
                except aiohttp.ClientError:
                    await asyncio.sleep(0.1)
            target = next(p for p in pages if p["type"] == "page")
            async with http_.ws_connect(target["webSocketDebuggerUrl"], max_msg_size=0) as ws:
                page = Page(ws)
                await page.call("Runtime.enable")
                await page.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=2, mobile=False)
                await page.call("Page.navigate", url=f"http://127.0.0.1:{PORT}/promo.html")
                for _ in range(300):
                    if await page.eval("window.promoReady === true"):
                        break
                    await asyncio.sleep(0.1)
                else:
                    raise RuntimeError("the page did not get ready")
                return await fn(page)
    finally:
        chrome.terminate()
        shutil.rmtree(profile, ignore_errors=True)


async def render_stills(times: list[float]) -> None:
    async def go(page):
        OUT.mkdir(parents=True, exist_ok=True)
        t_prev = 0.0
        for t in sorted(times):
            # step through the frames before it so events and timers run in order
            for f in range(int(t_prev * FPS), int(t * FPS)):
                await page.eval(f"renderFrame({f / FPS})")
            await page.eval(f"renderFrame({t})")
            (OUT / f"still_{t:05.2f}.png").write_bytes(await page.shot())
            print(f"still {t}")
            t_prev = t
    await with_page(go)


async def render_video(path: Path) -> dict:
    async def go(page):
        meta = await page.eval("window.PROMO")
        n = int(round(meta["duration"] * FPS))
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "png", "-i", "-",
                                "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high", "-level", "4.2",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        t0 = time.time()
        for f in range(n):
            await page.eval(f"renderFrame({f / FPS})")
            enc.stdin.write(await page.shot())
            if f % 120 == 0:
                el = time.time() - t0
                print(f"frame {f}/{n}  {el:.0f}s elapsed, ~{el / max(1, f) * (n - f):.0f}s left", flush=True)
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        return meta
    return await with_page(go)


# --------------------------------------------------------------------------------------------- #
# Music and sound effects
# --------------------------------------------------------------------------------------------- #
BPM = 120
BEAT = 60 / BPM
NOTE = {"A2": 110.0, "B2": 123.47, "C3": 130.81, "D3": 146.83, "E3": 164.81, "F2": 87.31, "G2": 98.0, "F3": 174.61,
        "G3": 196.0, "A3": 220.0, "B3": 246.94, "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, "G4": 392.0,
        "A4": 440.0, "B4": 493.88, "C5": 523.25, "D5": 587.33, "E5": 659.26, "G5": 783.99, "A5": 880.0}
# A minor: Am - F - C - G, two bars each
CHORDS = [("A2", ["A3", "C4", "E4"], ["A4", "C5", "E5", "C5"]), ("F2", ["F3", "A3", "C4"], ["A4", "C5", "F4", "C5"]),
          ("C3", ["G3", "C4", "E4"], ["G4", "C5", "E5", "C5"]), ("G2", ["G3", "B3", "D4"], ["G4", "B4", "D5", "B4"])]


def lowpass(x, cutoff, order=2):
    from scipy.signal import butter, sosfilt

    return sosfilt(butter(order, cutoff, "low", fs=SR, output="sos"), x)


def highpass(x, cutoff, order=2):
    from scipy.signal import butter, sosfilt

    return sosfilt(butter(order, cutoff, "high", fs=SR, output="sos"), x)


def saw(f, t, detune=0.0):
    ph = (f * (2 ** (detune / 1200))) * t
    return 2 * (ph - np.floor(ph + 0.5))


def add(buf, start, sig, gain=1.0, pan=0.0):
    i = int(round(start * SR))
    if i >= buf.shape[1] or i + len(sig) <= 0:
        return
    j = min(buf.shape[1], i + len(sig))
    s = sig[: j - i] * gain
    buf[0, i:j] += s * np.sqrt(0.5 * (1 - pan))
    buf[1, i:j] += s * np.sqrt(0.5 * (1 + pan))


def env(n, a, d, sustain=0.0, r=0.0, hold=None):
    t = np.arange(n) / SR
    out = np.where(t < a, t / max(a, 1e-4), sustain + (1 - sustain) * np.exp(-(t - a) / max(d, 1e-4)))
    if r and hold is not None:
        out *= np.clip(1 - (t - hold) / r, 0, 1) ** 2 * (t > hold) + (t <= hold)
    return out


def kick():
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 46 + 110 * np.exp(-t / 0.035)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.20)
    click = np.random.default_rng(1).standard_normal(n) * np.exp(-t / 0.004) * 0.25
    return np.tanh(1.6 * (body + highpass(click, 1500)))


def hat(open_=False, seed=2):
    n = int((0.25 if open_ else 0.06) * SR)
    t = np.arange(n) / SR
    noise = np.random.default_rng(seed).standard_normal(n)
    return highpass(noise, 7000, 4) * np.exp(-t / (0.08 if open_ else 0.018))


def clap(seed=3):
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    for k, dt in enumerate((0, 0.009, 0.018, 0.027)):
        i = int(dt * SR)
        e[i:] += np.exp(-(t[: n - i]) / (0.008 if k < 3 else 0.12))
    x = rng.standard_normal(n) * e
    from scipy.signal import butter, sosfilt

    return sosfilt(butter(2, [900, 4200], "band", fs=SR, output="sos"), x) * 1.6


def crash(seed=4, length=2.4):
    n = int(length * SR)
    t = np.arange(n) / SR
    return highpass(np.random.default_rng(seed).standard_normal(n), 3500, 2) * np.exp(-t / 0.7) * 0.5


def riser(length=1.6, seed=5):
    n = int(length * SR)
    t = np.arange(n) / SR
    noise = np.random.default_rng(seed).standard_normal(n)
    ramp = (t / length) ** 2
    sweep = np.sin(2 * np.pi * np.cumsum(300 + 1400 * (t / length) ** 2) / SR) * 0.15
    return (highpass(noise, 1200) * 0.5 + sweep) * ramp


def pluck(f, length=0.32):
    n = int(length * SR)
    t = np.arange(n) / SR
    x = 0.6 * np.sign(np.sin(2 * np.pi * f * t)) * 0.4 + 0.6 * saw(f, t)
    return lowpass(x, min(9000, f * 6)) * np.exp(-t / 0.11)


def music(duration: float) -> np.ndarray:
    n = int(duration * SR) + SR
    t = np.arange(n) / SR
    mix = {k: np.zeros((2, n)) for k in ("pad", "bass", "drums", "arp", "fx")}
    bar = 4 * BEAT
    # sections (seconds): what plays where
    drums_on = lambda s: 4.0 <= s < 56.0                                   # noqa: E731
    full = lambda s: (14.0 <= s < 24.0) or (30.0 <= s < 56.0)            # noqa: E731
    # pad: one chord per two bars, slow attack, low-passed, with a kick-driven duck
    for b in range(int(duration / bar) + 1):
        s0 = b * bar
        if s0 >= duration:
            break
        root, pad_notes, _ = CHORDS[(b // 2) % 4]
        length = 2 * bar if b % 2 == 0 else 0
        if not length:
            continue
        m = int(length * SR) + int(0.6 * SR)
        tt = np.arange(m) / SR
        sig = sum(saw(NOTE[p], tt, d) for p in pad_notes for d in (-7, 0, 7)) / 9
        cutoff = 900 if s0 < 4 else 1800
        sig = lowpass(sig, cutoff, 2) * env(m, 0.35, 10, 0.85, 0.5, length)
        if s0 >= 56:
            sig *= np.clip(1 - tt / 7.5, 0, 1) ** 1.5
        add(mix["pad"], s0, sig, 0.8, 0.0)
        # bass: eighth notes on the root
        if drums_on(s0):
            for k in range(int(length / (BEAT / 2))):
                st = s0 + k * BEAT / 2
                if st >= 56:
                    break
                bn = int(0.24 * SR)
                bt = np.arange(bn) / SR
                f = NOTE[root]
                bsig = (np.sin(2 * np.pi * f * bt) + 0.35 * saw(f, bt)) * np.exp(-bt / 0.16)
                add(mix["bass"], st, lowpass(np.tanh(1.4 * bsig), 900), 0.4 if k % 2 else 0.5)
    # arpeggio: sixteenths over the chord, panned
    for k in range(int(duration / (BEAT / 4))):
        st = k * BEAT / 4
        if st >= 56.5:
            break
        b = int(st // bar)
        _, _, arp = CHORDS[(b // 2) % 4]
        f = NOTE[arp[k % 4]] * (2 if (k // 8) % 4 == 3 and full(st) else 1)
        level = 0.2 if st < 4 else (0.3 if not full(st) else 0.36)
        add(mix["arp"], st, pluck(f), level, -0.45 if k % 2 else 0.45)
    # drums
    K, HH, CL = kick(), hat(), clap()
    for k in range(int(duration / BEAT)):
        st = k * BEAT
        if not drums_on(st):
            continue
        breakdown = 24.0 <= st < 30.0
        if not breakdown or k % 2 == 0:
            add(mix["drums"], st, K, 0.62)
        add(mix["drums"], st + BEAT / 2, HH, 0.22, 0.25)
        if full(st):
            add(mix["drums"], st + BEAT / 4, hat(seed=7 + k % 5), 0.09, -0.3)
            add(mix["drums"], st + 3 * BEAT / 4, hat(seed=11 + k % 3), 0.09, -0.3)
            if k % 2 == 1:
                add(mix["drums"], st, CL, 0.45, 0.05)
    for s in (14.0, 30.0, 36.0, 56.0):
        add(mix["fx"], s, crash(seed=int(s)), 0.55)
    for s in (14.0, 36.0):
        r = riser()
        add(mix["fx"], s - len(r) / SR, r, 0.35)
    # sidechain: duck the pad and bass after every kick
    duck = np.ones(n)
    for k in range(int(duration / BEAT)):
        st = k * BEAT
        if drums_on(st) and not (24.0 <= st < 30.0 and k % 2):
            i = int(st * SR)
            m = min(n - i, int(0.35 * SR))
            duck[i:i + m] = np.minimum(duck[i:i + m], 1 - 0.4 * np.exp(-np.arange(m) / SR / 0.09))
    for key in ("pad", "bass"):
        mix[key] *= duck
    out = sum(mix.values())
    return out[:, :int(duration * SR)]


def sound_effects(events: list[dict], duration: float) -> np.ndarray:
    n = int(duration * SR)
    out = np.zeros((2, n))
    rng = np.random.default_rng(9)
    for e in events:
        t, kind = e["t"], e["kind"]
        if kind == "click":
            m = int(0.05 * SR)
            tt = np.arange(m) / SR
            sig = (np.sin(2 * np.pi * 2300 * tt) * np.exp(-tt / 0.006) + highpass(rng.standard_normal(m), 3000) * np.exp(-tt / 0.003) * 0.4)
            add(out, t, sig, 0.35)
        elif kind in ("blip", "blip2"):
            m = int(0.09 * SR)
            tt = np.arange(m) / SR
            f = 1760 if kind == "blip" else 2349
            add(out, t, np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.025), 0.12)
        elif kind == "success":
            for k, f in enumerate((1318.5, 1760.0)):
                m = int(0.7 * SR)
                tt = np.arange(m) / SR
                sig = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt / 0.22)
                add(out, t + k * 0.09, sig, 0.16)
        elif kind == "whoosh":
            m = int(0.55 * SR)
            tt = np.arange(m) / SR
            bell = np.sin(np.pi * tt / (m / SR)) ** 2
            sig = lowpass(rng.standard_normal(m), 2500) * bell
            add(out, t - 0.2, sig, 0.10)
    return out


def write_audio(meta: dict, path: Path) -> None:
    import wave

    dur = meta["duration"]
    m = music(dur)
    fx = sound_effects(meta["sfx"], dur)
    mixdown = 0.7 * m + fx
    mixdown = np.tanh(mixdown * 1.1) / np.tanh(1.1)
    fade = int(1.2 * SR)
    mixdown[:, -fade:] *= np.linspace(1, 0, fade) ** 2
    mixdown /= max(1e-6, np.abs(mixdown).max()) / 0.9
    raw = path.with_suffix(".raw.wav")
    with wave.open(str(raw), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mixdown.T * 32767).astype("<i2").tobytes())
    # two-pass loudness normalisation to -14 LUFS (what X, Reddit and YouTube play at), true peak -1.5 dB
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(raw), "-af",
                          "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    j = json.loads(log[log.rindex("{"):log.rindex("}") + 1])
    af = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:"
          f"measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af", af, "-ar", str(SR), str(path)], check=True)
    raw.unlink()


def mux(video: Path, audio: Path, out: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(audio), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)], check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stills", help="comma-separated times to render as PNG instead of the video")
    p.add_argument("--audio-only", action="store_true", help="redo the sound on an existing render")
    a = p.parse_args()
    if not DEMO.exists():
        sys.exit("put the demo video at dev/demo/trailer.mp4 and run dev/site_assets.py first")
    build()
    server = serve()
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        if a.stills:
            asyncio.run(render_stills([float(x) for x in a.stills.split(",")]))
            return
        silent = OUT / "video_silent.mp4"
        meta_path = OUT / "meta.json"
        if a.audio_only and silent.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = asyncio.run(render_video(silent))
            meta_path.write_text(json.dumps(meta))
        write_audio(meta, OUT / "audio.wav")
        mux(silent, OUT / "audio.wav", OUT / "scenecut-promo.mp4")
        print(f"wrote {OUT / 'scenecut-promo.mp4'}")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
