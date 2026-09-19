"""Offline self-test.

Checks the HTTP routes and the exact payload contract the browser expects,
using synthetic activations so it runs without torch installed. If this
passes but the real run fails, the problem is the torch install, not the app.

Run:  python server/selftest.py
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import app as server            # noqa: E402
import inference                # noqa: E402
from labels import build_index  # noqa: E402

FAKE_CATEGORIES = (["tench", "goldfish"] * 200 +
                   ["sports car", "convertible", "limousine", "beach wagon",
                    "racer", "pickup", "minivan", "school bus", "fire engine",
                    "car wheel", "jeep", "trailer truck", "cab"] * 50)[:1000]


class FakeEngine:
    """Same payload shape as inference.Engine, random numbers inside."""

    def __init__(self):
        self.categories = FAKE_CATEGORIES
        self.vehicles, self.parts = build_index(self.categories)
        self.rng = np.random.default_rng(7)

    def analyse(self, img, top_k=6):
        crop, _ = inference.preprocess(img, 224)
        shapes = [(64, 112, 112), (64, 56, 56), (128, 28, 28),
                  (256, 14, 14), (512, 7, 7)]
        layers = []
        for (key, title, kind, note), shape in zip(
                inference.RESNET_STAGES, shapes):
            arr = np.clip(self.rng.normal(0.2, 0.6, shape).astype("float32"), 0, None)
            res = 32 if shape[-1] >= 32 else max(8, shape[-1])
            entry = {"id": key, "title": title, "kind": kind, "note": note}
            entry.update(inference.pack_maps(arr, show=16, res=res))
            layers.append(entry)

        vec = np.clip(self.rng.normal(0.3, 0.5, 512).astype("float32"), 0, None)
        key, title, kind, note = inference.RESNET_STAGES[5]
        entry = {"id": key, "title": title, "kind": kind, "note": note}
        entry.update(inference.pack_vector(vec))
        layers.append(entry)

        probs = self.rng.random(1000).astype("float32")
        probs[402] = 9.0
        probs = probs / probs.sum()
        order = np.argsort(-probs)[:top_k]
        key, title, kind, note = inference.RESNET_STAGES[6]
        layers.append({
            "id": key, "title": title, "kind": kind, "note": note,
            "shape": [1000],
            "top": [{"idx": int(i), "label": self.categories[int(i)],
                     "p": round(float(probs[int(i)]), 5)} for i in order],
        })

        return {
            "ok": True, "model": "fake",
            "input": {"kind": "image", "png": inference.png_data_url(crop),
                      "size": 224, "original": list(img.size)},
            "layers": layers,
            "verdict": inference.Engine._verdict(self, probs),
            "attribution": {"method": "grad-cam", "kind": "heatmap",
                            **inference.pack_heatmap(
                                np.abs(self.rng.normal(0, 1, (7, 7))).astype("float32"))},
            "timing_ms": {"forward": 99},
        }


def main():
    fake = FakeEngine()
    server.STATE.update({"engine": fake, "loading": False, "error": None})
    client = server.app.test_client()

    fails = []

    def check(name, cond, extra=""):
        print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else f"  {extra}"))
        if not cond:
            fails.append(name)

    print("\nroutes")
    r = client.get("/")
    check("GET /", r.status_code == 200 and b"NetSim3D" in r.data, r.status_code)
    for f in ("style.css", "main.js", "viz.js"):
        r = client.get(f"/static/{f}")
        check(f"GET /static/{f}", r.status_code == 200, r.status_code)
    r = client.get("/api/status")
    check("GET /api/status", r.status_code == 200 and r.get_json()["ready"])

    r = client.post("/api/analyze", data={})
    check("POST with no file is rejected", r.status_code == 400)
    r = client.post("/api/analyze", data={
        "image": (io.BytesIO(b"definitely not an image"), "x.jpg")})
    check("POST with junk is rejected", r.status_code == 400)

    print("\npayload contract")
    buf = io.BytesIO()
    Image.new("RGB", (900, 600), (70, 90, 140)).save(buf, format="JPEG")
    buf.seek(0)
    r = client.post("/api/analyze", data={"image": (buf, "car.jpg")})
    check("POST image returns 200", r.status_code == 200, r.status_code)
    p = r.get_json()
    check("ok flag", p.get("ok") is True)
    check("input png is a data url", p["input"]["png"].startswith("data:image/png;base64,"))
    check("7 layers captured", len(p["layers"]) == 7, len(p["layers"]))

    import base64
    convs = [l for l in p["layers"] if l["kind"] == "features"]
    check("5 conv stages", len(convs) == 5, len(convs))
    for l in convs:
        need = l["shown"] * l["res"] * l["res"]
        got = len(base64.b64decode(l["data"]))
        check(f"{l['id']} data length {got} == {need}", got == need)
        check(f"{l['id']} has shape/note/stats",
              all(k in l for k in ("shape", "note", "stats", "channels")))

    vec = [l for l in p["layers"] if l["kind"] == "vector"][0]
    check("vector values match shown", len(vec["values"]) == vec["shown"])
    out = [l for l in p["layers"] if l["kind"] == "output"][0]
    check("top-k has label and p", all("label" in t and "p" in t for t in out["top"]))
    check("probabilities descend",
          all(out["top"][i]["p"] >= out["top"][i + 1]["p"] for i in range(len(out["top"]) - 1)))
    at = p["attribution"]
    check("attribution heatmap present",
          len(base64.b64decode(at["data"])) == at["res"] ** 2)
    check("attribution names its method", bool(at.get("method")))
    check("input declares its kind", p["input"].get("kind") == "image")

    v = p["verdict"]
    check("verdict has the fields the UI reads",
          all(k in v for k in ("label", "p", "detail", "warning", "is_vehicle",
                               "vehicle_mass", "family", "vehicle", "part_hint")))

    print("\nanswering at the right level of confidence")
    eng = FakeEngine()
    probs = np.zeros(1000, dtype="float32")
    # the real distribution from the user's saloon photo: spread across seven
    # overlapping car classes, none of them convincing on its own
    spread = {"limousine": 0.274, "sports car": 0.170, "convertible": 0.158,
              "minivan": 0.123, "beach wagon": 0.063, "racer": 0.027}
    for name, val in spread.items():
        probs[eng.categories.index(name)] = val
    probs[0] = 1.0 - float(probs.sum())
    verdict = inference.Engine._verdict(eng, probs)
    check("a car spread thin across classes is still called a car",
          verdict["is_vehicle"] and verdict["family"] is not None)
    check("the headline is the confident family, not the shaky top-1",
          verdict["label"] == "Passenger car", verdict["label"])
    check("the specific class is still reported underneath",
          "limousine" in verdict["detail"], verdict["detail"])
    check("the user is warned that the fine class is a guess",
          bool(verdict["warning"]))

    probs2 = np.zeros(1000, dtype="float32")
    probs2[eng.categories.index("school bus")] = 0.86
    probs2[0] = 0.14
    v2 = inference.Engine._verdict(eng, probs2)
    check("a confident answer is reported plainly", v2["label"] == "Bus", v2["label"])
    check("a confident answer carries no warning", v2["warning"] is None)
    if v["vehicle"]:
        check("vehicle entry has pretty name and family",
              "pretty" in v["vehicle"] and "family" in v["vehicle"])

    size_kb = len(r.data) / 1024
    print(f"\n  payload size: {size_kb:.0f} KB")
    check("payload under 400 KB", size_kb < 400, f"{size_kb:.0f} KB")

    print("\nschema")
    for l in p["layers"]:
        check(f"{l['id']} has id/title/kind/note/shape",
              all(k in l for k in ("id", "title", "kind", "note", "shape")))
        check(f"{l['id']} kind is a known renderer kind",
              l["kind"] in {"features", "neurons", "vector", "tokens",
                            "sequence", "output", "scalar"})

    print("\nvehicle taxonomy")
    veh, parts = build_index(FAKE_CATEGORIES)
    check("vehicle classes found", len(veh) > 0, len(veh))
    check("part classes found", len(parts) > 0, len(parts))

    print()
    if fails:
        print(f"{len(fails)} check(s) failed: {fails}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
