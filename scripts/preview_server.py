"""Headless camera preview over HTTP (MJPEG).

Runs the camera in a background thread, exposes the latest frame as an MJPEG
stream so any browser on the network can watch live without an X server. Built
for the lighting/positioning loop where the operator needs to see what the
camera sees while adjusting hardware over SSH.

No new dependencies — stdlib `http.server` + threading + cv2 JPEG encoding.

Endpoints:
  GET  /              HTML page: live image + sidebar controls + status strip
  GET  /stream        multipart/x-mixed-replace MJPEG, one capture-thread shared
  GET  /snapshot.jpg  single JPEG of the latest frame
  GET  /control       apply ?exposure=us&gain=db; returns JSON with applied values
  GET  /control_state current camera settings (JSON) — used to seed the sliders
  GET  /stats         live frame stats (JSON): mean/std/sat/focus + fps + sensor
  GET  /save          ?lighting=&board=&note= — save full-res raw frame + meta
  GET  /healthz       "ok" + last frame age (ms)

Usage:
    .venv/bin/python scripts/preview_server.py
    .venv/bin/python scripts/preview_server.py --exposure 1500 --port 8080
    .venv/bin/python scripts/preview_server.py --max-width 800 --jpeg-quality 70
    .venv/bin/python scripts/preview_server.py --exposure-max 50000 --gain-max 24
    .venv/bin/python scripts/preview_server.py --save-dir data/captures/preview

Open http://<this-host>:8080/ in a browser. Drag sliders to tune exposure/gain
without restarting; type a lighting tag and click Save to dump a full-res PNG
plus meta JSON straight to the captures dir. Stop with Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pcb_inspection.camera import CameraConfig, create_camera

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MJPEG_BOUNDARY = "frame"


class FrameBuffer:
    """Holds the most recent frame in three forms: raw (full-res, for /save),
    JPEG (downscaled, for browser stream), and live stats (for the status bar).

    Single producer (CaptureThread), many consumers (HTTP handlers). The cv2
    encode is done in the producer so consumers only see the cheap bytes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._raw: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._stats: dict[str, float] | None = None
        self._captured_at: float = 0.0
        self._seq: int = 0

    def publish(self, raw: np.ndarray, jpeg: bytes, stats: dict[str, float]) -> None:
        with self._cond:
            self._raw = raw
            self._jpeg = jpeg
            self._stats = stats
            self._captured_at = time.monotonic()
            self._seq += 1
            self._cond.notify_all()

    def latest_jpeg(self) -> tuple[bytes | None, float, int]:
        with self._lock:
            return self._jpeg, self._captured_at, self._seq

    def latest_raw(self) -> tuple[np.ndarray | None, dict[str, float] | None, int]:
        with self._lock:
            return self._raw, dict(self._stats) if self._stats else None, self._seq

    def latest_stats(self) -> tuple[dict[str, float] | None, float, int]:
        with self._lock:
            stats = dict(self._stats) if self._stats else None
            return stats, self._captured_at, self._seq

    def wait_for_next(self, last_seq: int, timeout: float = 5.0) -> tuple[bytes | None, int]:
        """Block until a frame newer than last_seq arrives (or timeout)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > last_seq, timeout=timeout)
            return self._jpeg, self._seq


def compute_stats(gray_u8: np.ndarray) -> dict[str, float]:
    """Cheap per-frame indicators on the downscaled 8-bit gray image.

    Run on the preview-sized frame so it costs ~ms even on 25MP sensors.
    Mirrors capture_smoke.py semantics so 'sat' here matches that report.
    """
    flat = gray_u8.reshape(-1)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "sat_pct": float((flat >= 252).mean() * 100.0),
        "focus": float(cv2.Laplacian(gray_u8, cv2.CV_64F).var()),
    }


class Controls:
    """Pending + applied camera settings, mediated between HTTP and capture thread.

    HTTP handlers only ever call request(); the capture thread drains pending
    via take_pending() right before its next grab. This keeps cvsCam SDK calls
    on a single thread (the SDK is not promised to be reentrant) without any
    cross-thread cam.set_feature() races.
    """

    def __init__(self, exposure: float, gain: float, stream_width: int) -> None:
        self._lock = threading.Lock()
        self._applied: dict[str, float] = {
            "exposure_us": float(exposure),
            "gain_db": float(gain),
            "stream_width": float(stream_width),
        }
        self._pending: dict[str, float] = {}
        self._pending_reconnect: threading.Event | None = None

    def request(
        self,
        exposure: float | None = None,
        gain: float | None = None,
        stream_width: float | None = None,
    ) -> None:
        with self._lock:
            if exposure is not None:
                self._pending["exposure_us"] = float(exposure)
            if gain is not None:
                self._pending["gain_db"] = float(gain)
            if stream_width is not None:
                self._pending["stream_width"] = float(stream_width)

    def take_pending(self) -> dict[str, float]:
        with self._lock:
            out = self._pending
            self._pending = {}
            return out

    def mark_applied(self, applied: dict[str, float]) -> None:
        with self._lock:
            self._applied.update(applied)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._applied)

    def request_reconnect(self) -> threading.Event:
        """Ask the capture thread to close + re-enumerate + reopen the camera.

        Returns an Event the caller waits on; set when the capture thread
        finishes the cycle (success OR failure — check cam_info().error).
        """
        with self._lock:
            if self._pending_reconnect is None:
                self._pending_reconnect = threading.Event()
            return self._pending_reconnect

    def take_pending_reconnect(self) -> threading.Event | None:
        with self._lock:
            evt = self._pending_reconnect
            self._pending_reconnect = None
            return evt


class CaptureThread(threading.Thread):
    """Single owner of the camera. Grabs in a loop, encodes, publishes.

    All cvsCam SDK calls happen on this thread, including reconnect — the HTTP
    handler enqueues a reconnect request via Controls and waits on the event
    instead of touching the camera itself.
    """

    daemon = True

    def __init__(
        self,
        cam,
        buf: FrameBuffer,
        controls: Controls,
        max_width: int,
        jpeg_quality: int,
        target_fps: float,
        backend: str,
        device_index: int,
        cfg: CameraConfig,
        cam_info: dict | None = None,
    ) -> None:
        super().__init__(name="capture")
        self.cam = cam
        self.buf = buf
        self.controls = controls
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality
        self.min_interval = 1.0 / max(target_fps, 0.1)
        self.stop_event = threading.Event()
        self.backend = backend
        self.device_index = device_index
        self.cfg = cfg  # template; exposure/gain refreshed from controls on reconnect
        # Sliding window for actual delivered FPS (excludes paused/error frames).
        self._fps_lock = threading.Lock()
        self._fps_window: deque[float] = deque(maxlen=20)
        self._sensor_shape: tuple[int, int] | None = None  # (h, w) of raw frame
        self._info_lock = threading.Lock()
        self._cam_info: dict = dict(cam_info or {})

    def fps(self) -> float:
        with self._fps_lock:
            if len(self._fps_window) < 2:
                return 0.0
            span = self._fps_window[-1] - self._fps_window[0]
            return (len(self._fps_window) - 1) / span if span > 0 else 0.0

    def sensor_shape(self) -> tuple[int, int] | None:
        return self._sensor_shape

    def cam_info(self) -> dict:
        with self._info_lock:
            return dict(self._cam_info)

    def _maybe_reconnect(self) -> None:
        evt = self.controls.take_pending_reconnect()
        if evt is None:
            return
        logger.info("reconnect requested; closing current camera...")
        info: dict = {}
        try:
            try:
                self.cam.close()
            except Exception:
                logger.exception("close during reconnect failed; continuing")
            applied = self.controls.snapshot()
            new_cfg = CameraConfig(
                exposure_us=applied.get("exposure_us", self.cfg.exposure_us),
                gain=applied.get("gain_db", self.cfg.gain),
                pixel_format=self.cfg.pixel_format,
                width=self.cfg.width,
                height=self.cfg.height,
            )
            cam = create_camera(
                backend=self.backend, config=new_cfg, device_index=self.device_index,
            )
            info = cam.open() or {}
            self.cam = cam
            self.cfg = new_cfg
            self._sensor_shape = None  # repopulates on next grab
            with self._fps_lock:
                self._fps_window.clear()
            logger.info("reconnect ok: %s S/N %s", info.get("model"), info.get("serial"))
        except Exception as e:
            logger.exception("reconnect failed")
            info = {"error": str(e)}
        finally:
            with self._info_lock:
                self._cam_info = info
            evt.set()

    def _apply_pending(self) -> None:
        pending = self.controls.take_pending()
        if not pending:
            return
        applied: dict[str, float] = {}
        if "exposure_us" in pending:
            try:
                self.cam.set_feature("ExposureTime", pending["exposure_us"], "float")
                applied["exposure_us"] = pending["exposure_us"]
            except Exception:
                logger.exception("set ExposureTime=%s failed", pending["exposure_us"])
        if "gain_db" in pending:
            try:
                self.cam.set_feature("Gain", pending["gain_db"], "float")
                applied["gain_db"] = pending["gain_db"]
            except Exception:
                # Some camera models lack Gain — log once and forget.
                logger.warning("set Gain=%s failed (register may be unsupported)",
                               pending["gain_db"])
        if "stream_width" in pending:
            # Pure post-processing; no camera call. Min 80 to keep stats meaningful.
            self.max_width = max(80, int(pending["stream_width"]))
            applied["stream_width"] = float(self.max_width)
        if applied:
            self.controls.mark_applied(applied)

    def run(self) -> None:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        next_due = time.monotonic()
        while not self.stop_event.is_set():
            self._maybe_reconnect()
            self._apply_pending()
            try:
                raw = self.cam.grab()
            except Exception:
                logger.exception("grab failed; pausing 0.5s")
                time.sleep(0.5)
                continue
            self._sensor_shape = raw.shape[:2]

            gray_u8, display = self._prepare(raw)
            stats = compute_stats(gray_u8)

            ok, encoded = cv2.imencode(".jpg", display, encode_params)
            if not ok:
                logger.warning("JPEG encode failed; skipping frame")
                continue
            self.buf.publish(raw, encoded.tobytes(), stats)

            with self._fps_lock:
                self._fps_window.append(time.monotonic())

            # Pace so we don't pin the CPU on huge sensor frames.
            next_due += self.min_interval
            sleep = next_due - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # Fell behind — reset schedule rather than burst-catch-up.
                next_due = time.monotonic()

    def _prepare(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (gray_u8 for stats, BGR_u8 downscaled for JPEG stream)."""
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]
        if w > self.max_width:
            new_w = self.max_width
            new_h = int(h * (new_w / w))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return gray, bgr


def _index_html(host_hint: str, port: int, ranges: dict) -> bytes:
    # Layout: left sidebar (sliders + save form), main image, top status strip.
    # Sliders read /control_state on load, push debounced /control on input.
    # Status strip polls /stats every 500ms; "sat" turns red above 5%.
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>PCB camera preview</title>
<style>
 *{{box-sizing:border-box}}
 body{{margin:0;background:#111;color:#ddd;font-family:system-ui,sans-serif;
       font-size:13px;height:100vh;display:grid;
       grid-template-columns:300px 1fr;grid-template-rows:auto 1fr;
       grid-template-areas:'side status' 'side main'}}
 #status{{grid-area:status;background:#1a1a1a;border-bottom:1px solid #333;
         padding:6px 12px;display:flex;gap:18px;flex-wrap:wrap;
         font-variant-numeric:tabular-nums}}
 #status .k{{color:#888}} #status .v{{color:#eee}}
 #status .v.warn{{color:#f66}}
 #side{{grid-area:side;background:#181818;border-right:1px solid #333;
        padding:14px;overflow-y:auto;display:flex;flex-direction:column;gap:18px}}
 #main{{grid-area:main;background:#000;min-width:0;min-height:0;
        position:relative;overflow:auto}}
 /* fit (default): image scales to fill main area, letterboxed */
 #main.fit #stream{{display:block;width:100%;height:100%;object-fit:contain;
                    position:absolute;inset:0}}
 /* zoomed: image rendered at its natural pixel size × scale, scrollable */
 #main.zoom #stream{{display:block;object-fit:none;
                     transform-origin:top left;
                     image-rendering:pixelated}}
 /* grid overlay — sized + positioned by JS to match the rendered image area
    in both fit (letterboxed) and zoom (scrollable) modes. */
 #grid{{display:none;position:absolute;top:0;left:0;pointer-events:none;
        --gs:50px;
        background-image:
          repeating-linear-gradient(0deg,transparent 0 calc(var(--gs) - 1px),
            rgba(255,255,255,.28) calc(var(--gs) - 1px) var(--gs)),
          repeating-linear-gradient(90deg,transparent 0 calc(var(--gs) - 1px),
            rgba(255,255,255,.28) calc(var(--gs) - 1px) var(--gs))}}
 #grid.on{{display:block}}
 h2{{margin:0 0 8px;font-size:11px;letter-spacing:.08em;color:#888;
     text-transform:uppercase;font-weight:600}}
 .row{{display:flex;align-items:center;gap:8px;margin:6px 0}}
 .row label{{flex:0 0 70px;color:#aaa}}
 .row input[type=range]{{flex:1}}
 .row .v{{flex:0 0 60px;text-align:right;color:#fc6;font-variant-numeric:tabular-nums}}
 .row input[type=text],.row input[type=number],.row select{{flex:1;background:#222;
        border:1px solid #333;color:#ddd;padding:4px 6px;border-radius:3px;font:inherit}}
 .row input[type=number]::-webkit-inner-spin-button{{opacity:.5}}
 button{{width:100%;padding:8px;background:#2a5a8c;color:#fff;border:0;
        border-radius:3px;font:inherit;cursor:pointer;margin-top:6px}}
 button:hover{{background:#3a6aa0}}
 button:disabled{{background:#333;cursor:default}}
 #saveStatus{{margin-top:8px;font-size:11px;color:#888;min-height:14px;
              word-break:break-all}}
 #saveStatus.ok,#reconStatus.ok{{color:#7c6}}
 #saveStatus.err,#reconStatus.err{{color:#f66}}
 .hw{{font-size:12px;color:#bbb;line-height:1.5;margin-bottom:8px;
      padding:6px 8px;background:#222;border-radius:3px;
      font-variant-numeric:tabular-nums}}
 .hw .mono{{color:#888;font-size:11px}}
 .meta{{font-size:11px;color:#666;line-height:1.5}}
 .meta a{{color:#8af;text-decoration:none}}
</style></head><body>

<div id='status'>
 <span><span class='k'>cam</span> <span class='v' id='sCam'>—</span></span>
 <span><span class='k'>fps</span> <span class='v' id='sFps'>—</span></span>
 <span><span class='k'>age</span> <span class='v' id='sAge'>—</span>ms</span>
 <span><span class='k'>mean</span> <span class='v' id='sMean'>—</span></span>
 <span><span class='k'>std</span> <span class='v' id='sStd'>—</span></span>
 <span><span class='k'>sat</span> <span class='v' id='sSat'>—</span>%</span>
 <span><span class='k'>focus</span> <span class='v' id='sFocus'>—</span></span>
</div>

<aside id='side'>
 <div>
  <h2>Hardware</h2>
  <div class='hw' id='hwInfo'>—</div>
  <button id='btnReconnect' style='background:#3a3a3a'>Reconnect (rescan)</button>
  <div id='reconStatus' style='margin-top:8px;font-size:11px;color:#888;min-height:14px'></div>
 </div>

 <div>
  <h2>Camera</h2>
  <div class='row'><label>exposure</label>
   <input id='exp' type='range'/><span class='v' id='expv'>—</span></div>
  <div class='row'><label>gain</label>
   <input id='gain' type='range'/><span class='v' id='gainv'>—</span></div>
 </div>

 <div>
  <h2>Display</h2>
  <div class='row'><label>stream w</label>
   <input id='swidth' type='range'/><span class='v' id='swidthv'>—</span></div>
  <div class='row'><label>view</label>
   <select id='zoom'>
    <option value='fit' selected>Fit</option>
    <option value='0.5'>50%</option>
    <option value='1'>100% (1:1)</option>
    <option value='2'>200%</option>
    <option value='4'>400%</option>
   </select></div>
  <div class='row'><label>custom %</label>
   <input id='zoomCustom' type='number' min='10' max='1000' step='5' placeholder='e.g. 150'/></div>
  <div class='row'><label>grid</label>
   <input id='gridOn' type='checkbox' style='flex:0 0 auto;margin:0 4px 0 0'/>
   <input id='gridSp' type='number' min='4' max='400' step='2' value='50' style='flex:0 0 70px'/>
   <span style='color:#888;font-size:11px'>px</span></div>
 </div>

 <div>
  <h2>Save snapshot</h2>
  <div class='row'><label>lighting</label><input id='fLight' type='text' placeholder='dome | low_ring | coax'/></div>
  <div class='row'><label>board</label><input id='fBoard' type='text' placeholder='OK01'/></div>
  <div class='row'><label>note</label><input id='fNote' type='text' placeholder='optional'/></div>
  <button id='btnSave'>Save full-res PNG</button>
  <div id='saveStatus'></div>
 </div>

 <div class='meta'>
  preview · {host_hint}:{port}<br>
  <a href='/snapshot.jpg' target='_blank'>snapshot.jpg</a> ·
  <a href='/stats' target='_blank'>stats</a> ·
  <a href='/healthz' target='_blank'>healthz</a>
 </div>
</aside>

<main id='main' class='fit'><img id='stream' src='/stream' alt='live'/><div id='grid'></div></main>

<script>
const R = {json.dumps(ranges)};
const $ = id => document.getElementById(id);
const exp = $('exp'), expv = $('expv'), gain = $('gain'), gainv = $('gainv');
const swidth = $('swidth'), swidthv = $('swidthv');
const zoom = $('zoom'), main = $('main'), stream = $('stream');
const zoomCustom = $('zoomCustom'), grid = $('grid');
const gridOn = $('gridOn'), gridSp = $('gridSp');
exp.min = R.exposure_min; exp.max = R.exposure_max; exp.step = R.exposure_step;
gain.min = R.gain_min; gain.max = R.gain_max; gain.step = R.gain_step;
swidth.min = R.stream_width_min; swidth.max = R.stream_width_max;
swidth.step = R.stream_width_step;

fetch('/control_state').then(r => r.json()).then(s => {{
  exp.value = s.exposure_us; expv.textContent = Number(s.exposure_us).toFixed(0);
  gain.value = s.gain_db; gainv.textContent = Number(s.gain_db).toFixed(1);
  swidth.value = s.stream_width; swidthv.textContent = Number(s.stream_width).toFixed(0);
}});

let pending = null, timer = null;
function send() {{
  if (!pending) return;
  const q = new URLSearchParams(pending).toString();
  pending = null;
  fetch('/control?' + q).catch(() => {{}});
}}
function schedule(field, el, view, fmt, debounce_ms) {{
  el.addEventListener('input', () => {{
    view.textContent = fmt(el.value);
    pending = pending || {{}};
    pending[field] = el.value;
    if (timer) clearTimeout(timer);
    timer = setTimeout(send, debounce_ms);
  }});
}}
schedule('exposure', exp, expv, v => Number(v).toFixed(0), 50);
schedule('gain', gain, gainv, v => Number(v).toFixed(1), 50);
// Stream width is heavier (re-encode size changes), so debounce longer.
schedule('stream_width', swidth, swidthv, v => Number(v).toFixed(0), 200);

// View zoom mode — pure client-side CSS toggle, no server traffic.
// Set explicit pixel size in zoom mode so #main (overflow:auto) gives scrollbars
// when content exceeds the viewport. Custom % input takes precedence over
// the preset select; clearing it falls back to the select.
function applyZoom() {{
  let k = NaN;
  const cv = zoomCustom.value.trim();
  if (cv) k = parseFloat(cv) / 100;
  if (!Number.isFinite(k) || k <= 0) {{
    if (zoom.value === 'fit') {{
      main.className = 'fit';
      stream.style.width = stream.style.height = '';
      sizeFitGrid();
      return;
    }}
    k = parseFloat(zoom.value);
  }}
  main.className = 'zoom';
  if (stream.naturalWidth) {{
    const w = Math.round(stream.naturalWidth * k);
    const h = Math.round(stream.naturalHeight * k);
    stream.style.width = w + 'px';
    stream.style.height = h + 'px';
    grid.style.width = w + 'px';
    grid.style.height = h + 'px';
    grid.style.left = grid.style.top = '0px';
  }}
}}
// In fit mode the image is letterboxed by object-fit:contain. Compute the
// rendered image rect ourselves so the grid covers only the image, not the
// black bars on the side / top.
function sizeFitGrid() {{
  const nw = stream.naturalWidth, nh = stream.naturalHeight;
  if (!nw || !nh) {{
    grid.style.width = grid.style.height = '';
    grid.style.left = grid.style.top = '';
    return;
  }}
  const cw = main.clientWidth, ch = main.clientHeight;
  const scale = Math.min(cw / nw, ch / nh);
  const w = Math.round(nw * scale), h = Math.round(nh * scale);
  grid.style.width = w + 'px';
  grid.style.height = h + 'px';
  grid.style.left = Math.round((cw - w) / 2) + 'px';
  grid.style.top = Math.round((ch - h) / 2) + 'px';
}}
// Grid overlay — toggle visibility + screen-px spacing. Position/size handled
// by applyZoom (zoom mode) and sizeFitGrid (fit mode).
function applyGrid() {{
  grid.classList.toggle('on', gridOn.checked);
  const sp = Math.max(2, parseInt(gridSp.value) || 50);
  grid.style.setProperty('--gs', sp + 'px');
}}
zoom.addEventListener('change', () => {{ zoomCustom.value = ''; applyZoom(); }});
zoomCustom.addEventListener('input', applyZoom);
gridOn.addEventListener('change', applyGrid);
gridSp.addEventListener('input', applyGrid);
applyGrid();
// Re-apply on every JPEG load — natural size changes when stream-width slider
// moves, which also shifts the fit-mode letterbox.
stream.addEventListener('load', applyZoom);
// Window resize re-letterboxes the fit-mode image; zoom mode is unaffected.
window.addEventListener('resize', () => {{
  if (main.classList.contains('fit')) sizeFitGrid();
}});

// Status strip poll
async function pollStats() {{
  try {{
    const s = await (await fetch('/stats')).json();
    $('sCam').textContent = s.sensor_w && s.sensor_h
      ? `${{s.sensor_w}}×${{s.sensor_h}}` : '—';
    $('sFps').textContent = s.fps != null ? s.fps.toFixed(1) : '—';
    $('sAge').textContent = s.age_ms != null ? s.age_ms : '—';
    $('sMean').textContent = s.mean != null ? s.mean.toFixed(0) : '—';
    $('sStd').textContent = s.std != null ? s.std.toFixed(0) : '—';
    const sat = s.sat_pct;
    const satEl = $('sSat');
    satEl.textContent = sat != null ? sat.toFixed(1) : '—';
    satEl.classList.toggle('warn', sat != null && sat > 5);
    $('sFocus').textContent = s.focus != null ? s.focus.toFixed(0) : '—';
  }} catch (e) {{ /* swallow */ }}
}}
setInterval(pollStats, 500); pollStats();

// Hardware info + reconnect
const hwInfo = $('hwInfo');
function fmtCam(c, sensor) {{
  if (!c || !c.model) return '—';
  const sn = (c.serial || '').slice(-4);
  const dim = (sensor && sensor.length === 2) ? `${{sensor[1]}} × ${{sensor[0]}}` : '?';
  return `<b>${{c.model}}</b><br>S/N …${{sn}} · fw ${{c.firmware || '?'}}<br>` +
         `<span class='mono'>${{dim}} · ${{c.mac || ''}}</span>`;
}}
async function refreshHw() {{
  try {{
    const j = await (await fetch('/camera/info')).json();
    hwInfo.innerHTML = fmtCam(j.camera, j.sensor);
  }} catch (e) {{ /* swallow */ }}
}}
refreshHw();
const btnRe = $('btnReconnect'), reSt = $('reconStatus');
btnRe.addEventListener('click', async () => {{
  btnRe.disabled = true; reSt.textContent = 'rescanning camera...'; reSt.className = '';
  try {{
    const r = await fetch('/camera/reconnect');
    const j = await r.json();
    if (j.ok) {{
      reSt.textContent = `connected: ${{j.camera.model}} S/N ${{j.camera.serial}}`;
      reSt.className = 'ok';
      hwInfo.innerHTML = fmtCam(j.camera, j.sensor);
      // Sensor shape resets on reconnect; first frame after open repopulates it.
      setTimeout(refreshHw, 1500);
      // MJPEG stream auto-resumes since the same /stream endpoint keeps publishing.
    }} else {{
      reSt.textContent = 'failed: ' + (j.error || r.status); reSt.className = 'err';
    }}
  }} catch (e) {{ reSt.textContent = 'error: ' + e; reSt.className = 'err'; }}
  btnRe.disabled = false;
}});

// Save
const btn = $('btnSave'), st = $('saveStatus');
btn.addEventListener('click', async () => {{
  btn.disabled = true; st.textContent = 'saving...'; st.className = '';
  const q = new URLSearchParams({{
    lighting: $('fLight').value, board: $('fBoard').value, note: $('fNote').value,
  }}).toString();
  try {{
    const r = await fetch('/save?' + q);
    const j = await r.json();
    if (j.ok) {{ st.textContent = 'saved: ' + j.file; st.className = 'ok'; }}
    else {{ st.textContent = 'error: ' + (j.error || r.status); st.className = 'err'; }}
  }} catch (e) {{ st.textContent = 'error: ' + e; st.className = 'err'; }}
  btn.disabled = false;
}});
</script></body></html>""".encode("utf-8")


_TAG_OK = re.compile(r"[^a-z0-9_+-]+")


def _sanitize_tag(s: str, max_len: int = 32) -> str:
    """Lower, replace non-[a-z0-9_+-] with '_', collapse, trim, length-cap."""
    s = _TAG_OK.sub("_", s.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len]


def make_handler(buf: FrameBuffer, controls: Controls, capture: CaptureThread,
                 ranges: dict, save_dir: Path, host_hint: str, port: int):
    class Handler(BaseHTTPRequestHandler):
        # Suppress per-request log lines — too noisy for an MJPEG stream.
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            url = urlsplit(self.path)
            path = url.path
            if path in ("/", "/index.html"):
                self._send_bytes(200, "text/html; charset=utf-8",
                                 _index_html(host_hint, port, ranges))
            elif path == "/snapshot.jpg":
                jpeg, _, _ = buf.latest_jpeg()
                if jpeg is None:
                    self.send_error(503, "No frame yet")
                    return
                self._send_bytes(200, "image/jpeg", jpeg)
            elif path == "/stream":
                self._stream(buf)
            elif path == "/control":
                self._control(parse_qs(url.query))
            elif path == "/control_state":
                body = json.dumps(controls.snapshot()).encode("utf-8")
                self._send_bytes(200, "application/json", body)
            elif path == "/stats":
                self._stats()
            elif path == "/save":
                self._save(parse_qs(url.query))
            elif path == "/camera/info":
                self._json(200, {"camera": capture.cam_info(),
                                 "sensor": list(capture.sensor_shape() or ())})
            elif path == "/camera/reconnect":
                self._reconnect()
            elif path == "/healthz":
                _, captured_at, seq = buf.latest_jpeg()
                age_ms = int((time.monotonic() - captured_at) * 1000) if captured_at else -1
                body = f"ok seq={seq} last_frame_age_ms={age_ms}\n".encode()
                self._send_bytes(200, "text/plain; charset=utf-8", body)
            else:
                self.send_error(404)

        def _control(self, qs: dict[str, list[str]]) -> None:
            try:
                exposure = float(qs["exposure"][0]) if "exposure" in qs else None
                gain = float(qs["gain"][0]) if "gain" in qs else None
                stream_width = float(qs["stream_width"][0]) if "stream_width" in qs else None
            except (ValueError, IndexError):
                self.send_error(400, "exposure/gain/stream_width must be numeric")
                return
            if exposure is not None:
                exposure = max(ranges["exposure_min"], min(ranges["exposure_max"], exposure))
            if gain is not None:
                gain = max(ranges["gain_min"], min(ranges["gain_max"], gain))
            if stream_width is not None:
                stream_width = max(ranges["stream_width_min"],
                                   min(ranges["stream_width_max"], stream_width))
            controls.request(exposure=exposure, gain=gain, stream_width=stream_width)
            body = json.dumps({"requested": {
                "exposure_us": exposure, "gain_db": gain, "stream_width": stream_width,
            }}).encode()
            self._send_bytes(200, "application/json", body)

        def _stats(self) -> None:
            stats, captured_at, seq = buf.latest_stats()
            sensor = capture.sensor_shape()  # (h, w) or None
            age_ms = int((time.monotonic() - captured_at) * 1000) if captured_at else None
            payload: dict[str, object] = {
                "seq": seq,
                "fps": round(capture.fps(), 2),
                "age_ms": age_ms,
                "sensor_w": sensor[1] if sensor else None,
                "sensor_h": sensor[0] if sensor else None,
            }
            if stats:
                payload.update({k: round(v, 3) for k, v in stats.items()})
            self._send_bytes(200, "application/json", json.dumps(payload).encode("utf-8"))

        def _save(self, qs: dict[str, list[str]]) -> None:
            raw, stats, seq = buf.latest_raw()
            if raw is None:
                self._json(200, {"ok": False, "error": "no frame yet"})
                return
            lighting = _sanitize_tag(qs.get("lighting", [""])[0])
            board = _sanitize_tag(qs.get("board", [""])[0])
            note = qs.get("note", [""])[0][:200]  # kept verbatim (in JSON only)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            parts = [p for p in (lighting, board, ts) if p]
            stem = "__".join(parts) if parts else ts
            save_dir.mkdir(parents=True, exist_ok=True)
            png_path = save_dir / f"{stem}.png"
            json_path = save_dir / f"{stem}.json"
            try:
                if not cv2.imwrite(str(png_path), raw):
                    raise RuntimeError("cv2.imwrite returned False")
                meta = {
                    "schema_version": "1",
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "seq": seq,
                    "lighting": lighting or None,
                    "board": board or None,
                    "note": note or None,
                    "camera": controls.snapshot(),
                    "image": {
                        "file": png_path.name,
                        "shape": list(raw.shape),
                        "dtype": str(raw.dtype),
                    },
                    "stats": stats,
                    "source": "preview_server",
                }
                json_path.write_text(json.dumps(meta, indent=2))
            except Exception as e:
                logger.exception("save failed")
                self._json(500, {"ok": False, "error": str(e)})
                return
            logger.info("saved %s (raw %s, stats=%s)", png_path.name, raw.shape, stats)
            try:
                rel = png_path.relative_to(Path.cwd())
            except ValueError:
                rel = png_path
            self._json(200, {"ok": True, "file": str(rel)})

        def _json(self, status: int, payload: dict) -> None:
            self._send_bytes(status, "application/json",
                             json.dumps(payload).encode("utf-8"))

        def _reconnect(self) -> None:
            evt = controls.request_reconnect()
            if not evt.wait(timeout=15.0):
                self._json(504, {"ok": False, "error": "reconnect timeout (15s)"})
                return
            info = capture.cam_info()
            if "error" in info:
                self._json(500, {"ok": False, "error": info["error"]})
                return
            self._json(200, {"ok": True, "camera": info,
                             "sensor": list(capture.sensor_shape() or ())})

        def _send_bytes(self, status: int, ctype: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self, buf: FrameBuffer) -> None:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last_seq = -1
            try:
                while True:
                    jpeg, last_seq = buf.wait_for_next(last_seq)
                    if jpeg is None:
                        continue
                    chunk = (
                        f"--{MJPEG_BOUNDARY}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n"
                    ).encode("ascii")
                    self.wfile.write(chunk)
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                # Browser closed the tab — normal.
                return

    return Handler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MJPEG preview server for the camera")
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default: all interfaces)")
    p.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    p.add_argument("--device", type=int, default=0, help="Camera device index")
    p.add_argument("--backend", default="crevis", choices=("crevis", "mock", "auto"))
    p.add_argument("--exposure", type=float, default=1500.0, help="Exposure (us)")
    p.add_argument("--gain", type=float, default=0.0, help="Gain (dB)")
    p.add_argument("--pixel-format", default="Mono8")
    p.add_argument("--width", type=int, default=None, help="Sensor ROI width (default sensor max)")
    p.add_argument("--height", type=int, default=None, help="Sensor ROI height (default sensor max)")
    p.add_argument("--max-width", type=int, default=1024,
                   help="Downscale frames above this width before sending. Default 1024.")
    p.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1-100. Default 80.")
    p.add_argument("--fps", type=float, default=10.0, help="Target preview FPS. Default 10.")
    p.add_argument("--exposure-min", type=float, default=100.0, help="Slider lower bound (us)")
    p.add_argument("--exposure-max", type=float, default=50000.0, help="Slider upper bound (us)")
    p.add_argument("--exposure-step", type=float, default=100.0, help="Slider step (us)")
    p.add_argument("--gain-min", type=float, default=0.0, help="Slider lower bound (dB)")
    p.add_argument("--gain-max", type=float, default=24.0, help="Slider upper bound (dB)")
    p.add_argument("--gain-step", type=float, default=0.5, help="Slider step (dB)")
    p.add_argument("--stream-width-min", type=int, default=320,
                   help="Stream width slider lower bound (px)")
    p.add_argument("--stream-width-max", type=int, default=2400,
                   help="Stream width slider upper bound (px). Cap to keep WiFi happy.")
    p.add_argument("--stream-width-step", type=int, default=80,
                   help="Stream width slider step (px)")
    p.add_argument(
        "--save-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "captures" / "preview",
        help="Where /save dumps full-res PNG + meta JSON",
    )
    return p.parse_args()


def _local_hint() -> str:
    """Best-effort 'what URL should I open' hint for the log line."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostname()


def main() -> None:
    args = parse_args()

    cfg = CameraConfig(
        exposure_us=args.exposure,
        gain=args.gain,
        pixel_format=args.pixel_format,
        width=args.width,
        height=args.height,
    )
    cam = create_camera(backend=args.backend, config=cfg, device_index=args.device)
    cam_info = cam.open() or {}

    buf = FrameBuffer()
    controls = Controls(exposure=args.exposure, gain=args.gain, stream_width=args.max_width)
    ranges = {
        "exposure_min": args.exposure_min, "exposure_max": args.exposure_max,
        "exposure_step": args.exposure_step,
        "gain_min": args.gain_min, "gain_max": args.gain_max, "gain_step": args.gain_step,
        "stream_width_min": args.stream_width_min,
        "stream_width_max": args.stream_width_max,
        "stream_width_step": args.stream_width_step,
    }
    capture = CaptureThread(
        cam, buf, controls, args.max_width, args.jpeg_quality, args.fps,
        backend=args.backend, device_index=args.device, cfg=cfg, cam_info=cam_info,
    )
    capture.start()

    host_hint = _local_hint()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(buf, controls, capture, ranges, args.save_dir, host_hint, args.port),
    )
    logger.info("Preview ready: http://%s:%d/  (Ctrl-C to stop)", host_hint, args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        capture.stop_event.set()
        capture.join(timeout=2.0)
        server.server_close()
        try:
            cam.close()
        except Exception:
            logger.exception("camera close failed")


if __name__ == "__main__":
    main()
