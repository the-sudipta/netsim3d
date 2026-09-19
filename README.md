# NetSim3D

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/the-sudipta/netsim3d)

Upload a photo, watch a real neural network read it in 3D, stage by stage,
and see the class it lands on.

Every glowing tile in the simulation is an actual activation computed from
your image by PyTorch. Nothing is pre-rendered and nothing is faked. If you
upload a different car you get a different pattern, because you are looking
at the tensors themselves.

---

## Run it

**Windows**

1. Unzip the folder anywhere (Desktop is fine).
2. Double-click **run.bat**.
3. Wait. The first run installs PyTorch and Flask (about 300 MB, once), then
   the browser opens by itself at `http://127.0.0.1:8765`.
4. Drop in a photo of a car.

**Python version matters.** Use regular CPython 3.9 to 3.14. A
*free-threaded* build (the no-GIL one, whose wheels are tagged `cp313t`
rather than `cp313`) will not work: PyTorch publishes no wheels for it, so
pip tries to compile numpy and pillow from source and fails. The launcher
detects this, refuses that interpreter, and tells you what to install. If you
have both, it picks the usable one and leaves your 3.13 alone.

**macOS / Linux**

```bash
chmod +x run.sh
./run.sh
```

The first image you upload also downloads the model weights, about 100 MB.
After that the whole thing runs offline except for the 3D library (see
*Offline use* below).

To check the install without a model or internet: `run.bat test`
(or `./run.sh test`). It exercises the server and the data contract and
prints a pass/fail list.

---

## Run it on GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/the-sudipta/netsim3d)

A codespace is a Linux machine attached to this repository, with a terminal
and port forwarding. It runs the project exactly as a laptop does, because it
is the same thing: a localhost, just someone else's. Nothing to install, and
`.devcontainer/` is already committed, so the machine arrives ready.

### Start it

**1. Create the codespace.** On the repository page: **Code** → **Codespaces**
→ **Create codespace on main**. Or click the badge above.

**2. Wait for the build.** Three to five minutes the first time. The terminal
installs the CPU build of PyTorch and pre-downloads the ResNet-50 weights, so
your first upload is instant rather than stalling for 100 MB while someone
watches. It finishes with:

```
Ready. Start it with:   ./run.sh
```

**3. Start the server.**

```
bash run.sh
```

Use `bash run.sh` rather than `./run.sh`. If the repository was pushed from
Windows, the Unix executable bit was lost in transit and `./run.sh` fails with
`Permission denied`. `bash run.sh` always works. To fix it permanently, run
this once from any machine and push:

```
git update-index --chmod=+x run.sh
```

**4. Open it.** A notification offers port 8765: click **Open in Browser**. If
you miss it, use the **Ports** tab beside the terminal and click the globe
icon next to 8765. The URL looks like:

```
https://cautious-guide-xxxxxxx-8765.app.github.dev
```

**5. Upload a car photo.** Same app, same simulation, running on GitHub's
machine instead of yours.

### Share the link with someone else

Forwarded ports are **private** by default, so anyone you send the URL to gets
a GitHub sign-in page instead of the app.

In the **Ports** tab, right-click 8765 → **Port Visibility** → **Public**.
Then copy the address and send it. Anyone with the link can use it, no GitHub
account needed, for as long as the codespace is running.

Visibility is per codespace, so check it again after a restart.

### How long it stays active

**A codespace stops after 30 minutes of inactivity.** That is the default; you
can set anything from 5 minutes to 240 minutes (4 hours) under your profile →
**Settings** → **Codespaces** → **Default idle timeout**. The new value applies
to codespaces you create afterwards, not to ones that already exist.

What counts as activity matters more than the number:

| Resets the 30-minute timer | Does not |
|---|---|
| Typing or moving the mouse in the editor | Leaving the browser tab open and idle |
| Any terminal output, including the server's request log | A shared port with nobody using it |
| Someone loading the page or uploading an image, because Flask logs every request to the terminal | A background process that prints nothing |

That third row is the useful one. While a visitor is actually using the demo,
each request prints a line in the terminal running `run.sh`, which resets the
idle timer. **So the codespace stays up as long as someone is using it**, and
shuts down about half an hour after the last person stops. You do not need to
sit there keeping it awake during a demo.

Two consequences worth knowing:

- You are billed for compute for the whole time it is **running**, whether or
  not you are using it. Stop it when you are finished rather than leaving it
  idling for 30 minutes.
- Stopping does **not** delete it. Your files, the installed packages and the
  cached model weights all survive, and restarting takes seconds instead of
  minutes. Stopped codespaces are kept for 30 days by default, then deleted
  automatically. That retention period is configurable from 0 to 30 days in
  the same settings page.

### Stopping, restarting and deleting

Stop it from the command palette (**Codespaces: Stop Current Codespace**) or
at [github.com/codespaces](https://github.com/codespaces).

Restart it from the same page, or through **Code** → **Codespaces** on the
repository. Then `bash run.sh` again.

**Storage is charged for as long as a codespace exists, stopped or not.**
Delete ones you have finished with rather than just stopping them.

### What it costs

Personal GitHub accounts include **120 Codespaces core-hours and 15 GB of
storage per month**. Core-hours are CPU cores multiplied by runtime, so a
2-core machine consumes 2 core-hours per real hour: roughly **60 hours of
runtime a month, free**. This project is comfortable on 2 cores.

### Pulling in later changes

The codespace holds its own clone. After pushing changes from elsewhere:

```
git pull
```

If `.devcontainer/` changed, rebuild the container: command palette →
**Codespaces: Rebuild Container**.

### What this is not

The URL dies when the codespace stops, so this is a session you share for an
afternoon, not a permanent demo link for a paper. For that, the model has to
run in the visitor's browser (ONNX Runtime Web on GitHub Pages) or on a host
that keeps a Python process alive.

### Continuous checks

`.github/workflows/check.yml` runs the project's own test suites on every
push: compiles every Python file, runs the payload-contract self-test, parses
both browser modules, and verifies that every DOM id and CSS class the front
end uses actually exists. No model and no GPU needed, so it finishes in under
a minute and stays inside the free Actions allowance. Results appear under the
**Actions** tab.

### Codespace troubleshooting

| What you see | What to do |
|---|---|
| `./run.sh: Permission denied` | Use `bash run.sh`. Permanent fix: `git update-index --chmod=+x run.sh`, commit, push |
| `bad interpreter: No such file or directory` | `run.sh` picked up Windows line endings. `.gitattributes` prevents this; to repair a file that already has them, `sed -i 's/\r$//' run.sh` |
| Your visitor sees a GitHub sign-in page | Port 8765 is still private. Ports tab → right-click → Port Visibility → Public |
| Port 8765 never appears | The server did not start. Read the terminal where you ran `bash run.sh` |
| The build failed | Command palette → **Codespaces: Rebuild Container**, or run `bash .devcontainer/setup.sh` by hand and read the error |
| It stopped while you were mid-demo | 30 minutes passed with no terminal output. Restart it and raise the idle timeout in Settings → Codespaces |
| Out of core-hours | 120 per month on a personal account. Delete stopped codespaces, they keep consuming the storage quota |

## What you are looking at

| Stage | What is on screen | What is happening in the model |
|---|---|---|
| Image in | Your photo, centre-cropped to 224×224 | Resize, crop, normalise with the ImageNet mean and standard deviation |
| conv1 + relu | 16 tiles, and a yellow window sliding over your photo | One 7×7 convolution, stride 2, 64 filters. The sliding window is the kernel; each tile is one filter's response to the whole image |
| stage 1 | 16 tiles at 56×56 | Two residual blocks. Corners, short curves, texture |
| stage 2 | 16 tiles at 28×28 | 128 filters. Wheel arcs, grille slats, window frames |
| stage 3 | 16 tiles at 14×14 | 256 filters. Parts and their arrangement, almost no fine detail left |
| stage 4 | 16 tiles at 7×7 | 512 filters. Whole-object evidence |
| global average pool | A comb of thin bars | Each 7×7 map collapses to one number. The image is now 512 numbers |
| Decision | Horizontal bars with class names | One dot product per class, then softmax. Amber bar is the answer |
| Where it looked | Heat map burning back onto your photo | Grad-CAM. Bright pixels are the ones that drove the decision |
| Whole network | Wide shot of the full stack | Orbit and zoom freely |

Only the 16 strongest feature maps per stage are drawn, out of 64 to 512. The
inspector panel tells you how many exist and how many are firing. Drawing all
512 would be a grey mush at flythrough speed.

## The answer, and how much to trust it

The model is **ResNet-50** with the improved V2 weights: 80.8% top-1 on
ImageNet against 69.8% for the ResNet-18 this used to ship with. It knows
about forty vehicle types, from sports car and limousine to articulated lorry
and tram.

Three things make the answer steadier than a plain forward pass:

- **Each model's own preprocessing.** The V2 recipe resizes to 232 before
  cropping to 224, not 256. Using the wrong number costs accuracy for nothing.
- **Three views, averaged.** The centre crop, its mirror image, and a squashed
  full-frame view that rescues photos where the car is not centred. Two extra
  forward passes, noticeably steadier probabilities. The simulation always
  shows the centre crop, because that is the view whose activations are
  captured.
- **Answering at the confidence the model actually has.** ImageNet's car
  classes overlap badly. A saloon routinely lands on limousine at 27% with
  sports car at 17% and convertible at 16%, which reads as a wrong answer even
  though the model is around 70% sure it is a passenger car and merely unsure
  which kind. So the headline is the **family** when the family is confident,
  the closest specific class is named underneath, and you get a warning when
  the top two are close.

Upload something that is not a vehicle and it says so, and tells you if it
only found a part (a wheel, a grille) rather than a whole car.

`NETSIM_MODEL=resnet18` switches back to the smaller, faster, less accurate
model when you want quicker iteration.

## Training it on your own car types

ImageNet's classes are the ones ImageNet has. If you need *sedan / SUV /
hatchback / pickup*, or specific makes and models, nothing but labelled
examples will fix that. `finetune.py` does it, and the simulation picks the
result up automatically.

```
data/
  sedan/      img001.jpg img002.jpg ...
  suv/        ...
  hatchback/  ...
  pickup/     ...
```

```
.venv\Scripts\python.exe finetune.py          (Windows)
.venv/bin/python finetune.py                   (macOS, Linux)
```

It starts from the ImageNet weights, trains the last block and the head with
sensible augmentation, prints validation accuracy every epoch, and saves the
best one to `weights/custom.pt`. Then:

```
set NETSIM_MODEL=custom && run.bat          (Windows)
NETSIM_MODEL=custom ./run.sh                (macOS, Linux)
```

The whole simulation works unchanged on your classes, and the header reports
your validation accuracy so nobody mistakes a 64% model for a 95% one.

About 150 images per class is a floor, 500+ is comfortable, and `--full`
trains every layer once you have plenty of data. Public sets worth knowing
about: Stanford Cars, CompCars, VMMRdb. Check each licence before using one
in a publication.

---

## Controls

- **Play / Pause**, **Back / Forward** step one stage at a time
- Click any chapter chip to jump to that stage
- Drag the timeline to scrub; the 3D scene follows exactly
- Speed: 0.35× is the one to use when explaining it to someone
- Zoom moves about 4% per wheel notch and eases in over roughly 0.4 seconds.
  It covers the same share of the range whether you are beside a tile or
  looking at the whole network, so it never lurches. Large jumps are what the
  chapter chips and **Re-centre camera** are for
- **Glow**: off / low / medium / high, default medium. Controls bloom and
  exposure together. Recording and projectors usually look better one step
  lower, because video compression exaggerates bloom
- Drag to orbit, scroll to zoom, right-drag to pan. The camera is yours at any
  time, including mid-run. While the simulation keeps playing, the pivot drifts
  to follow the stage being narrated, so you keep your own angle and distance
  without being left facing empty space. **Re-centre camera** puts it back on
  the scripted path
- **Hover anything** for an explanation: a feature tile tells you which filter
  it is and what bright pixels mean, a pooled bar gives its value and index, a
  class bar gives its rank and probability, the heat map explains itself
- Keyboard: space, left, right, and `r` to replay
- **Record video** captures the run to a `.webm` file in your downloads

`.webm` plays in any browser and in VLC. For PowerPoint or a journal
submission, convert it:

```bash
ffmpeg -i netsim3d-simulation.webm -c:v libx264 -crf 18 -pix_fmt yuv420p sim.mp4
```

---

## Use your own model

This is the part that matters if you want to show *your* network rather than
a stock one. Two edits in `server/inference.py`.

**1. Describe the stages you want on screen.** The first item in each tuple
is the real module name inside your model. Print `dict(model.named_modules()).keys()`
to see yours.

```python
MY_STAGES = [
    ("features.0",  "first conv",  "conv",   "What to tell the viewer here."),
    ("features.4",  "block 2",     "conv",   "..."),
    ("features.8",  "block 3",     "conv",   "..."),
    ("avgpool",     "pooled",      "vector", "..."),
    ("classifier",  "classifier",  "output", "..."),
]

MODELS["mymodel"] = dict(stages=MY_STAGES, cam_layer="features.8", size=224)
```

Kinds: `conv` for anything shaped (C, H, W), `vector` for a 1-D tensor,
`output` for the logits. Use as many `conv` stages as you like; the timeline,
camera path and chapter list rebuild themselves.

**2. Load your weights** in `Engine.load`, replacing the torchvision lines:

```python
import torch
self.model = MyNet(num_classes=7)
self.model.load_state_dict(torch.load("weights/mynet.pt", map_location="cpu"))
self.model.eval().to(self.device)
self.categories = ["hatchback", "sedan", "suv", "pickup", "van", "bus", "truck"]
```

Then run with `set NETSIM_MODEL=mymodel` before `run.bat`, or edit
`DEFAULT_MODEL`.

If your classes are not ImageNet vehicles, either extend `VEHICLE_LABELS` in
`server/labels.py` with your own label names, or ignore the vehicle panel and
read the plain top-k list.

Environment variables: `NETSIM_MODEL`, `NETSIM_PORT` (default 8765),
`NETSIM_HOST`, `NETSIM_OPEN=0` to stop it opening a browser, and
`NETSIM_PYTHON` to force a specific interpreter for the `.venv`.

---

## Offline use

The 3D library loads from a CDN at page load. To run with no internet at all,
download these two folders once and drop them next to the app:

- `https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js`
  → `static/vendor/three.module.js`
- `https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/`
  → `static/vendor/addons/`

Then change the import map in `static/index.html` to:

```json
{ "imports": { "three": "/static/vendor/three.module.js",
               "three/addons/": "/static/vendor/addons/" } }
```

The model weights cache in `~/.cache/torch/hub/checkpoints` after the first
download, so they are only fetched once either way.

---

## If something goes wrong

| What you see | What it means |
|---|---|
| "Python was not found" | Install Python 3.10+ and tick *Add python.exe to PATH* |
| "The headers or library files could not be found for zlib" while building pillow, paths containing `313t` | You are on a free-threaded Python. Install regular CPython 3.12 (`winget install Python.Python.3.12`), delete `.venv`, run again |
| "A dependency has no prebuilt wheel" | Same cause, different symptom. Nothing is compiled from source on purpose; use Python 3.12 |
| Install dies resolving numpy's build dependencies | An old copy of `run.bat` that put `--index-url https://download.pytorch.org/whl/cpu` before the base packages. That flag replaces PyPI instead of adding to it. Current launcher installs base packages first |
| PyTorch install fails some other way | No internet, a corporate proxy, or low disk space. It needs about 1 GB free |
| Want to see which interpreters you have | `run.bat pythons` |
| Anything specific to Codespaces | see the Codespace troubleshooting table above |
| Codespace: `bad interpreter: No such file or directory` | `run.sh` has Windows line endings. `.gitattributes` prevents this; if a file already has them, `sed -i 's/\r$//' run.sh` |
| "Could not load the 3D library" | The CDN is blocked. See *Offline use* |
| "Cannot reach the Python server" | The run window was closed. Start `run.bat` again |
| "Model failed to load" with a connection error | The weights need internet once |
| Black screen, no tiles | Very old GPU or WebGL disabled. Try Chrome or Edge |
| Scene went black after orbiting or zooming | Fixed. Earlier builds culled back faces and started fog at 165 units while the stack reaches 158, so orbiting behind or zooming out erased everything. Zoom, fog and clipping are now derived from the real size of the network, and every surface is double sided. If you still manage it, press **Re-centre camera** |
| Recording button does nothing | Firefox and Safari are patchy here. Chrome or Edge work |
| Everything is slow | Normal on an old CPU. The forward pass is about 100 ms; it is the 3D that costs. Lower the browser window size |

---

## What is in the box

```
run.bat / run.sh        one-click launcher: venv, install, start, open browser
requirements.txt        flask, pillow, numpy (torch installed by the launcher)
server/app.py           Flask server, two endpoints: /api/status, /api/analyze
server/inference.py     forward hooks, activation packing, Grad-CAM
server/labels.py        the vehicle taxonomy and families
finetune.py             train on your own classes, then simulate that model
server/selftest.py      offline check of the routes and the data contract
server/check_python.py  interpreter suitability check used by the launchers
.devcontainer/          GitHub Codespaces setup: Python, CPU torch, weights
.github/workflows/      runs the test suites on every push
static/index.html       page shell and the Three.js import map
static/viz.js           the 3D scene and the timeline
static/main.js          upload, panels, playback, video recording
static/style.css        interface styling
samples/                put test images here
```

Python owns the model and every number on screen. The browser is only the
display surface, because a real-time flythrough needs a GPU and the browser
is the one place every machine already has one. There is no Node, no npm and
no build step.

---

## Honest limits

- 16 maps per stage, resized to at most 32×32 and normalised per stage. Good
  for showing structure, not for reading exact values.
- The threads between stages are illustrative. A real 3×3 convolution has a
  specific receptive field per unit; drawing all of them is unreadable, so
  the threads suggest connectivity rather than trace it exactly.
- The kernel window sliding over your photo is drawn at 7×7 steps for
  legibility. The real conv1 slides 112 times across and 112 down.
- Grad-CAM comes from stage 4 only, which is the standard choice.
- The three averaged views improve the answer, but only the centre crop's
  activations are drawn. Saying otherwise would misrepresent which tensors
  you are looking at.
- Hover tooltips cover the drawn objects. The 16 maps you can see are
  hoverable; the 496 you cannot see are not.
- One image at a time, one user at a time. It is a local explainer, not a
  service.