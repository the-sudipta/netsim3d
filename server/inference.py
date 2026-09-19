"""Runs the image through the network and packs every intermediate tensor
into a compact JSON payload the browser can animate.

Nothing here knows about 3D. It answers one question per layer: what did this
layer actually output for this image, small enough to send over HTTP.
"""

from __future__ import annotations

import base64
import io
import os
import time

import numpy as np
from PIL import Image

from labels import FAMILY_PRETTY, build_index

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Which tensors we show, and what to tell the user about each one.
# Module names below are the real attribute names inside the torchvision model.
# ---------------------------------------------------------------------------

RESNET_STAGES = [
    ("relu", "conv1 + relu", "features",
     "One 7x7 convolution, stride 2. Each tile is one of 64 filters sweeping "
     "the whole image. Bright means the filter found what it looks for: an "
     "edge at some angle, a colour boundary, a patch of texture."),
    ("layer1", "stage 1", "features",
     "Two residual blocks at 56x56. Filters now respond to corners, short "
     "curves and repeated texture rather than single edges."),
    ("layer2", "stage 2", "features",
     "Halved again to 28x28, 128 filters. Parts start to appear: wheel arcs, "
     "grille slats, window frames, tyre tread."),
    ("layer3", "stage 3", "features",
     "14x14, 256 filters. Almost no fine detail left. What survives here is "
     "evidence about parts and their arrangement."),
    ("layer4", "stage 4", "features",
     "7x7, 512 filters. Each cell sees most of the image. This is the layer "
     "Grad-CAM reads to draw the heat map back onto your photo."),
    ("avgpool", "global average pool", "vector",
     "Every 7x7 map collapses to a single number. The image is now one "
     "512-number fingerprint. Spatial position has been thrown away."),
    ("fc", "classifier", "output",
     "One dot product per class, then softmax. The tall bar is the answer."),
]

MODELS = {
    # ResNet-50 with the improved V2 recipe: 80.8% top-1 on ImageNet against
    # 69.8% for ResNet-18. On fine-grained vehicle classes the gap is wider
    # still, which is the whole reason it is the default.
    "resnet50": dict(stages=RESNET_STAGES, cam_layer="layer4", size=224,
                     arch="resnet50"),
    "resnet18": dict(stages=RESNET_STAGES, cam_layer="layer4", size=224,
                     arch="resnet18"),
    # Your own fine-tuned weights, written by finetune.py. Same stage names,
    # so the entire simulation works unchanged on your own classes.
    "custom": dict(stages=RESNET_STAGES, cam_layer="layer4", size=224,
                   arch=None, checkpoint="weights/custom.pt"),
}

DEFAULT_MODEL = "resnet50"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Averaging a few views of the same photo is the cheapest accuracy available:
# no retraining, two extra forward passes, and much steadier probabilities on
# awkward crops. The squashed full-frame view rescues photos where the subject
# is not centred, so it carries a little less weight than the proper crops.
TTA_WEIGHTS = {"centre": 1.0, "flip": 1.0, "full": 0.6}


# ---------------------------------------------------------------------------
# Tensor -> payload packing (pure numpy, no torch; unit-testable on its own)
# ---------------------------------------------------------------------------

def _resize_map(chan: np.ndarray, res: int) -> np.ndarray:
    """Resize one HxW float map to res x res."""
    if chan.shape == (res, res):
        return chan
    img = Image.fromarray(chan.astype(np.float32), mode="F")
    return np.asarray(img.resize((res, res), Image.BILINEAR), dtype=np.float32)


def pack_maps(maps: np.ndarray, show: int = 16, res: int = 32):
    """Select the most active channels, resize, normalise, encode as base64.

    maps: float array of shape (C, H, W).
    Returns a dict describing what to draw, with `data` holding
    show*res*res uint8 values (one byte per activation sample).
    """
    if maps.ndim != 3:
        raise ValueError(f"expected (C,H,W), got {maps.shape}")
    c, h, w = maps.shape
    pos = np.clip(maps, 0, None)
    strength = pos.reshape(c, -1).mean(axis=1)
    order = np.argsort(-strength)          # strongest channel first
    keep = order[: min(show, c)]

    small = np.stack([_resize_map(pos[i], res) for i in keep])
    hi = float(np.percentile(small, 99.5)) if small.size else 0.0
    if hi <= 1e-8:
        hi = 1.0
    norm = np.clip(small / hi, 0.0, 1.0)
    bytes_ = (norm * 255.0 + 0.5).astype(np.uint8)

    live = float((pos > 1e-6).mean())
    return {
        "shape": [int(c), int(h), int(w)],
        "shown": int(len(keep)),
        "res": int(res),
        "channels": [int(i) for i in keep],
        "data": base64.b64encode(bytes_.tobytes()).decode("ascii"),
        "stats": {
            "mean": round(float(pos.mean()), 4),
            "max": round(float(maps.max()), 4),
            "alive": round(live, 4),
        },
    }


def pack_vector(vec: np.ndarray, show: int = 96):
    """Pack a 1-D activation vector (the pooled fingerprint)."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    hi = float(np.percentile(np.abs(vec), 99.5)) or 1.0
    idx = np.argsort(-np.abs(vec))[: min(show, vec.size)]
    idx = np.sort(idx)
    vals = np.clip(vec[idx] / hi, -1.0, 1.0)
    return {
        "shape": [int(vec.size)],
        "shown": int(idx.size),
        "index": [int(i) for i in idx],
        "values": [round(float(v), 4) for v in vals],
        "stats": {"mean": round(float(vec.mean()), 4),
                  "max": round(float(vec.max()), 4),
                  "alive": round(float((vec > 1e-6).mean()), 4)},
    }


def pack_heatmap(cam: np.ndarray, res: int = 28):
    cam = np.clip(np.asarray(cam, dtype=np.float32), 0, None)
    cam = _resize_map(cam, res)
    hi = float(cam.max()) or 1.0
    b = (np.clip(cam / hi, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return {"res": res, "data": base64.b64encode(b.tobytes()).decode("ascii")}


def _to_tensor(crop: Image.Image, mean, std):
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    norm = (arr - mean) / std
    return np.transpose(norm, (2, 0, 1))[None]              # (1,3,S,S)


def preprocess(img: Image.Image, size: int, resize: int = None,
               mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Resize-shortest-side then centre crop, matching torchvision eval.

    `resize` comes from the weights' own transform when available: the V2
    recipes resize to 232 rather than 256, and using the wrong one costs real
    accuracy for free.
    """
    img = img.convert("RGB")
    short = int(resize or round(size * 256 / 224))
    w, h = img.size
    scale = short / min(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                     Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    crop = img.crop((left, top, left + size, top + size))
    return crop, _to_tensor(crop, mean, std)


def png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# The engine (needs torch)
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, name: str = DEFAULT_MODEL):
        if name not in MODELS:
            raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
        self.name = name
        self.spec = MODELS[name]
        self.model = None
        self.device = None
        self.accuracy = None
        self.crop_size = self.spec["size"]
        self.resize_size = None
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD
        self.categories = []
        self.vehicles = {}
        self.parts = {}
        self._torch = None

    # -- loading ------------------------------------------------------------
    def load(self, log=print):
        import torch
        from torchvision import models as tvm

        self._torch = torch
        torch.set_grad_enabled(True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = self.spec.get("checkpoint")
        if ckpt_path and os.path.exists(os.path.join(ROOT, ckpt_path)):
            self._load_custom(os.path.join(ROOT, ckpt_path), tvm, log)
        elif ckpt_path:
            raise FileNotFoundError(
                f"No fine-tuned weights at {ckpt_path}. Run finetune.py first, "
                "or unset NETSIM_MODEL to use the stock model.")
        else:
            arch = self.spec.get("arch") or self.name
            log(f"  loading {arch} (first run downloads the weights, ~100 MB)")
            builder = getattr(tvm, arch)
            try:
                weight_enum = tvm.get_model_weights(arch).DEFAULT
            except Exception:                    # noqa: BLE001  older torchvision
                enum_name = "".join(p.capitalize() for p in arch.split("_"))
                weight_enum = getattr(tvm, f"{enum_name}_Weights").DEFAULT
            self.model = builder(weights=weight_enum)
            self.categories = list(weight_enum.meta["categories"])
            self.accuracy = weight_enum.meta.get("_metrics", {}).get(
                "ImageNet-1K", {}).get("acc@1")
            self._read_transform(weight_enum, log)

        self.model.eval()
        self.model.to(self.device)

        # A guessed module name silently produces a payload with a stage
        # missing, so check them against the model that actually loaded.
        named = dict(self.model.named_modules())
        absent = [k for k, *_ in self.spec["stages"] if k not in named]
        if absent:
            log(f"  warning: these stages are not in this model and will be "
                f"skipped: {absent}")

        self.vehicles, self.parts = build_index(self.categories)
        acc = f", {self.accuracy:.1f}% top-1" if self.accuracy else ""
        log(f"  ready: {len(self.categories)} classes, "
            f"{len(self.vehicles)} of them vehicles{acc}, on {self.device}")
        return self

    def _read_transform(self, weight_enum, log):
        """Use the weights' own preprocessing rather than assuming 256/224."""
        try:
            t = weight_enum.transforms()
            first = lambda v: v[0] if isinstance(v, (list, tuple)) else v  # noqa: E731
            self.crop_size = int(first(getattr(t, "crop_size", 224)))
            self.resize_size = int(first(getattr(t, "resize_size", 256)))
            self.mean = np.array(getattr(t, "mean", IMAGENET_MEAN), dtype=np.float32)
            self.std = np.array(getattr(t, "std", IMAGENET_STD), dtype=np.float32)
        except Exception as exc:                 # noqa: BLE001
            log(f"  using default preprocessing ({exc})")

    def _load_custom(self, path, tvm, log):
        """Load weights written by finetune.py."""
        torch = self._torch
        log(f"  loading your fine-tuned model from {os.path.relpath(path, ROOT)}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.categories = list(ckpt["classes"])
        arch = ckpt.get("arch", "resnet50")
        self.model = getattr(tvm, arch)(weights=None,
                                        num_classes=len(self.categories))
        self.model.load_state_dict(ckpt["state_dict"])
        self.accuracy = ckpt.get("val_acc")
        self.crop_size = int(ckpt.get("crop_size", 224))
        self.resize_size = int(ckpt.get("resize_size", 256))
        self.mean = np.array(ckpt.get("mean", IMAGENET_MEAN), dtype=np.float32)
        self.std = np.array(ckpt.get("std", IMAGENET_STD), dtype=np.float32)
        log(f"  {len(self.categories)} of your own classes: "
            f"{', '.join(self.categories[:6])}"
            f"{' ...' if len(self.categories) > 6 else ''}")

    # -- the actual run -----------------------------------------------------
    def analyse(self, img: Image.Image, top_k: int = 6):
        torch = self._torch
        if self.model is None:
            raise RuntimeError("engine not loaded")

        size = self.crop_size
        crop, batch = preprocess(img, size, self.resize_size, self.mean, self.std)
        x = torch.from_numpy(batch).to(self.device)

        grabbed = {}
        handles = []
        named = dict(self.model.named_modules())
        wanted = [s[0] for s in self.spec["stages"]]
        for key in wanted:
            mod = named.get(key)
            if mod is None:
                continue

            def make_hook(k):
                def hook(_m, _i, out):
                    grabbed[k] = out
                return hook
            handles.append(mod.register_forward_hook(make_hook(key)))

        cam_key = self.spec["cam_layer"]
        t0 = time.time()
        try:
            # Primary view: this is the one the simulation shows, and the one
            # whose graph Grad-CAM needs, so it runs first and with gradients.
            logits = self.model(x)
            probs = torch.softmax(logits[0], dim=0) * TTA_WEIGHTS["centre"]
            weight_sum = TTA_WEIGHTS["centre"]

            # Extra views only sharpen the answer. Hooks come off first so the
            # captured activations stay those of the primary view.
            for hd in handles:
                hd.remove()
            handles = []
            with torch.no_grad():
                flipped = _to_tensor(crop.transpose(Image.FLIP_LEFT_RIGHT),
                                     self.mean, self.std)
                full = _to_tensor(img.convert("RGB").resize((size, size),
                                                            Image.BICUBIC),
                                  self.mean, self.std)
                for arr, key in ((flipped, "flip"), (full, "full")):
                    try:
                        extra = self.model(torch.from_numpy(arr).to(self.device))
                        probs = probs + torch.softmax(extra[0], dim=0) * TTA_WEIGHTS[key]
                        weight_sum += TTA_WEIGHTS[key]
                    except Exception as exc:      # noqa: BLE001
                        print(f"  extra view {key} skipped: {exc}")
            probs = probs / weight_sum
            best = int(torch.argmax(probs).item())

            cam_map = None
            cam_act = grabbed.get(cam_key)
            if cam_act is not None and cam_act.requires_grad:
                try:
                    self.model.zero_grad(set_to_none=True)
                    grads = torch.autograd.grad(logits[0, best], cam_act,
                                                retain_graph=False)[0]
                    weights = grads[0].mean(dim=(1, 2), keepdim=True)
                    cam = (weights * cam_act[0]).sum(dim=0)
                    cam_map = torch.relu(cam).detach().cpu().numpy()
                except Exception as exc:          # noqa: BLE001
                    print("  grad-cam skipped:", exc)
        finally:
            for hd in handles:
                hd.remove()
        forward_ms = int((time.time() - t0) * 1000)

        # ---- pack layers -------------------------------------------------
        layers = []
        for key, title, kind, note in self.spec["stages"]:
            if key not in grabbed:
                continue
            t = grabbed[key].detach()
            entry = {"id": key, "title": title, "kind": kind, "note": note}
            if kind == "features":
                arr = t[0].cpu().numpy()
                show = 16 if arr.shape[0] >= 16 else arr.shape[0]
                res = 32 if arr.shape[-1] >= 32 else int(arr.shape[-1])
                res = max(res, 8)
                entry.update(pack_maps(arr, show=show, res=res))
            elif kind == "vector":
                entry.update(pack_vector(t[0].cpu().numpy().reshape(-1)))
            else:
                vals = probs.detach().cpu().numpy()
                order = np.argsort(-vals)[:top_k]
                entry.update({
                    "shape": [int(vals.size)],
                    "top": [{"idx": int(i),
                             "label": self.categories[int(i)],
                             "p": round(float(vals[int(i)]), 5)} for i in order],
                })
            layers.append(entry)

        payload = {
            "ok": True,
            "model": self.name,
            "views": len(TTA_WEIGHTS),
            "accuracy": self.accuracy,
            "input": {"kind": "image", "png": png_data_url(crop),
                      "size": size, "original": list(img.size)},
            "layers": layers,
            "verdict": self._verdict(probs.detach().cpu().numpy()),
            "device": str(self.device),
            "timing_ms": {"forward": forward_ms},
        }
        if cam_map is not None:
            payload["attribution"] = {"method": "grad-cam", "kind": "heatmap",
                                      **pack_heatmap(cam_map)}
        return payload

    # -- "what car is it?" --------------------------------------------------
    def _verdict(self, probs: np.ndarray):
        """Answer at the level of confidence the model actually has.

        ImageNet splits cars across seven overlapping classes, so a photo of a
        saloon routinely lands on limousine at 27% with sports car at 17%.
        Reporting only the top-1 there is misleading: the model is 70% sure it
        is a passenger car and genuinely unsure which kind. So the headline is
        the family when the family is confident, with the fine class beneath
        it, and the split is stated outright when the top two are close.
        """
        order = np.argsort(-probs)
        top = int(order[0])

        fam_mass, fam_best = {}, {}
        for i, (fam, _pretty) in self.vehicles.items():
            p = float(probs[i])
            fam_mass[fam] = fam_mass.get(fam, 0.0) + p
            if p > fam_best.get(fam, (None, 0.0))[1]:
                fam_best[fam] = (i, p)
        veh_mass = float(sum(fam_mass.values()))

        family = None
        if fam_mass:
            name = max(fam_mass, key=fam_mass.get)
            family = {"name": name,
                      "pretty": FAMILY_PRETTY.get(name, name.title()),
                      "p": round(fam_mass[name], 5)}

        best_veh = None
        for i in order[:50]:
            i = int(i)
            if i in self.vehicles:
                fam, pretty = self.vehicles[i]
                best_veh = {"idx": i, "label": self.categories[i],
                            "family": fam, "pretty": pretty,
                            "p": round(float(probs[i]), 5)}
                break

        part_hint = None
        for i in order[:10]:
            i = int(i)
            if i in self.parts:
                part_hint = self.parts[i]
                break

        is_vehicle = bool(best_veh and veh_mass >= 0.12)
        runner = int(order[1]) if probs.size > 1 else top
        close = float(probs[runner]) > float(probs[top]) * 0.6

        if is_vehicle and family and family["p"] >= 0.45:
            label = family["pretty"]
            detail = (f"{family['p'] * 100:.0f}% sure it is a "
                      f"{family['pretty'].lower()}; closest specific class is "
                      f"{best_veh['label']} at {best_veh['p'] * 100:.0f}%")
        elif is_vehicle:
            label = best_veh["pretty"]
            detail = (f"closest ImageNet class {best_veh['label']} at "
                      f"{best_veh['p'] * 100:.0f}%")
        else:
            label = self.categories[top]
            detail = (f"no whole vehicle found, but it did see {part_hint}"
                      if part_hint else f"{probs[top] * 100:.0f}% confident")

        warning = None
        if is_vehicle and close and best_veh:
            warning = (f"The model is split between {self.categories[top]} and "
                       f"{self.categories[runner]}. ImageNet's car classes "
                       f"overlap heavily, so treat the exact type as a guess.")
        elif float(probs[top]) < 0.18:
            warning = ("Low confidence throughout. Try a photo where the "
                       "subject fills more of the frame.")

        return {
            "label": label,
            "p": round(float(probs[top]), 5),
            "detail": detail,
            "warning": warning,
            "top_label": self.categories[top],
            "is_vehicle": is_vehicle,
            "vehicle_mass": round(veh_mass, 5),
            "family": family,
            "vehicle": best_veh,
            "part_hint": part_hint,
        }
