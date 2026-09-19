"""Fine-tune the classifier on your own images, then simulate it.

The stock model knows ImageNet's ~40 vehicle classes, which overlap badly:
a photo of a saloon lands somewhere between limousine, sports car and
convertible. If you need real answers for specific types, the honest fix is
training on labelled examples of those types. This does that, and writes a
checkpoint the simulation picks up automatically.

Put your images in folders named after the classes you want:

    data/
      sedan/       img001.jpg img002.jpg ...
      suv/         ...
      hatchback/   ...
      pickup/      ...

Then:

    .venv\\Scripts\\python.exe finetune.py            (Windows)
    .venv/bin/python finetune.py                     (macOS, Linux)

and run the app with NETSIM_MODEL=custom. Everything else, the flythrough,
Grad-CAM, the tooltips, works unchanged on your classes.

Roughly 150 images per class is a sensible floor. 500+ is comfortable.
Public sets worth knowing about: Stanford Cars (196 models), CompCars, and
VMMRdb. Check each one's licence before using it in a publication.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="Fine-tune on your own images.")
    ap.add_argument("--data", default="data", help="folder of class folders")
    ap.add_argument("--out", default="weights/custom.pt")
    ap.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--full", action="store_true",
                    help="train the whole network rather than the last block "
                         "and the head. Slower, better with lots of data.")
    args = ap.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, random_split
        from torchvision import datasets, models, transforms
    except ImportError:
        sys.exit("PyTorch is not installed in this environment. "
                 "Run run.bat once first, then use .venv's python.")

    data_dir = os.path.join(ROOT, args.data)
    if not os.path.isdir(data_dir):
        sys.exit(f"No folder at {data_dir}. Create it with one subfolder per "
                 f"class, each holding that class's images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Augmentation is what stops a small dataset from being memorised. Scale
    # and flip cover the ways the same car appears in different photos;
    # colour jitter covers lighting and paint.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.25, 0.25, 0.20, 0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(args.size * 256 / 224)),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full = datasets.ImageFolder(data_dir, transform=train_tf)
    classes = full.classes
    if len(classes) < 2:
        sys.exit(f"Found {len(classes)} class folder(s) in {data_dir}. "
                 "Two or more are needed.")

    counts = {c: 0 for c in classes}
    for _, y in full.samples:
        counts[classes[y]] += 1
    print(f"\n  {len(full)} images across {len(classes)} classes")
    for c in classes:
        flag = "  <- thin, expect weak results" if counts[c] < 60 else ""
        print(f"    {c:<24} {counts[c]:>5}{flag}")

    n_val = max(1, int(len(full) * args.val_split))
    gen = torch.Generator().manual_seed(1234)
    train_set, val_set = random_split(full, [len(full) - n_val, n_val], generator=gen)
    val_set.dataset = datasets.ImageFolder(data_dir, transform=eval_tf)

    train_ld = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                          num_workers=0)
    val_ld = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                        num_workers=0)

    # Start from ImageNet features. With a few hundred images per class that
    # is the difference between something usable and noise.
    weights = models.get_model_weights(args.arch).DEFAULT
    model = getattr(models, args.arch)(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    if not args.full:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(("layer4", "fc"))
        print("\n  training the last block and the head only "
              "(use --full to train everything)")

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    print(f"  device {device}, {args.epochs} epochs\n")
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(xb)
            seen += len(xb)
        sched.step()

        model.eval()
        right = n = 0
        with torch.no_grad():
            for xb, yb in val_ld:
                pred = model(xb.to(device)).argmax(1).cpu()
                right += int((pred == yb).sum())
                n += len(yb)
        acc = 100.0 * right / max(1, n)
        print(f"  epoch {epoch:>2}/{args.epochs}  loss {total / max(1, seen):.3f}"
              f"  val {acc:5.1f}%  ({time.time() - t0:.0f}s)")

        if acc >= best:
            best = acc
            out = os.path.join(ROOT, args.out)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            torch.save({
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": round(acc, 2),
                "crop_size": args.size,
                "resize_size": int(args.size * 256 / 224),
                "mean": mean, "std": std,
            }, out)

    print(f"\n  best validation accuracy {best:.1f}%")
    print(f"  saved to {args.out}")
    print("\n  Run the simulation on it:")
    print("    Windows :  set NETSIM_MODEL=custom && run.bat")
    print("    mac/Linux: NETSIM_MODEL=custom ./run.sh")
    if best < 70:
        print("\n  Under 70% usually means too few images per class, classes "
              "that overlap visually, or both. More data beats more epochs.")


if __name__ == "__main__":
    main()
