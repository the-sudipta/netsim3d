"""NetSim3D — local server.

Python does everything that matters: it holds the model, runs the forward
pass, extracts every intermediate tensor and serves it. The browser is only
the display surface, because a real-time 3D flythrough has to be rendered by
a GPU and the browser is the one place every machine already has one.

Run:  python server/app.py        (or just use run.bat / run.sh)
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
import traceback
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402
from PIL import Image  # noqa: E402

import inference  # noqa: E402

HOST = os.environ.get("NETSIM_HOST", "127.0.0.1")
PORT = int(os.environ.get("NETSIM_PORT", "8765"))
MODEL = os.environ.get("NETSIM_MODEL", inference.DEFAULT_MODEL)
AUTO_OPEN = os.environ.get("NETSIM_OPEN", "1") != "0"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

STATE = {"engine": None, "error": None, "loading": True}
LOCK = threading.Lock()


def load_engine():
    try:
        STATE["engine"] = inference.Engine(MODEL).load()
    except Exception as exc:                      # noqa: BLE001
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        print("\n  model load failed:", STATE["error"])
        if "urlopen" in str(exc) or "Connection" in str(exc):
            print("  the first run needs internet once, to fetch the weights.")
        traceback.print_exc()
    finally:
        STATE["loading"] = False


@app.get("/")
def index():
    return send_from_directory(os.path.join(ROOT, "static"), "index.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(os.path.join(ROOT, "static"), name)


@app.get("/api/status")
def status():
    eng = STATE["engine"]
    return jsonify({
        "loading": STATE["loading"],
        "error": STATE["error"],
        "model": MODEL,
        "ready": eng is not None,
        "classes": len(eng.categories) if eng else 0,
        "vehicle_classes": len(eng.vehicles) if eng else 0,
        "accuracy": getattr(eng, "accuracy", None) if eng else None,
        "stages": [{"id": s[0], "title": s[1], "kind": s[2]}
                   for s in inference.MODELS[MODEL]["stages"]],
    })


@app.post("/api/analyze")
def analyze():
    if STATE["loading"]:
        return jsonify({"ok": False, "error": "Model is still loading. "
                                              "Try again in a few seconds."}), 503
    eng = STATE["engine"]
    if eng is None:
        return jsonify({"ok": False, "error": STATE["error"] or
                        "Model unavailable."}), 500

    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify({"ok": False, "error": "No image in the request."}), 400
    try:
        raw = file.read()
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:                             # noqa: BLE001
        return jsonify({"ok": False,
                        "error": "That file is not a readable image. "
                                 "Use JPG, PNG or WEBP."}), 400

    try:
        with LOCK:                                # one forward pass at a time
            payload = eng.analyse(img)
    except Exception as exc:                      # noqa: BLE001
        traceback.print_exc()
        return jsonify({"ok": False,
                        "error": f"{type(exc).__name__}: {exc}"}), 500
    return jsonify(payload)


def banner():
    print("\n" + "=" * 62)
    print("  NetSim3D — layer-by-layer network simulation")
    print("=" * 62)
    print(f"  model : {MODEL}")
    print(f"  open  : http://{HOST}:{PORT}")
    print("  stop  : Ctrl+C in this window")
    print("=" * 62 + "\n")


def open_browser_when_ready():
    url = f"http://{HOST}:{PORT}"
    for _ in range(120):
        if not STATE["loading"]:
            break
        time.sleep(0.5)
    time.sleep(0.6)
    try:
        webbrowser.open(url)
    except Exception:                             # noqa: BLE001
        print(f"  open {url} in your browser")


if __name__ == "__main__":
    banner()
    threading.Thread(target=load_engine, daemon=True).start()
    if AUTO_OPEN:
        threading.Thread(target=open_browser_when_ready, daemon=True).start()
    app.run(host=HOST, port=PORT, threaded=True, debug=False,
            use_reloader=False)
