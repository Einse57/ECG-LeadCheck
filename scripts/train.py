"""Train the interchange classifier. usage: train.py DATA_DIR OUT.pt [--folds 1-8] [--epochs 40] [--seed 0]"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from leadcheck.data import finish, load  # noqa: E402
from leadcheck.model import LeadNet  # noqa: E402
from leadcheck.swaps import CLASSES, apply  # noqa: E402


class Train(Dataset):
    def __init__(self, X, idx, epoch_size):
        self.X, self.idx, self.n = X, idx, epoch_size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng()
        x = self.X[self.idx[rng.integers(len(self.idx))]]
        y = 0 if rng.random() < 0.4 else int(rng.integers(1, len(CLASSES)))
        x = finish(apply(x, CLASSES[y], rng))
        x = x * rng.uniform(0.8, 1.25) + rng.normal(0, rng.uniform(0, 0.03), x.shape)
        t = np.arange(x.shape[1]) / 100.0
        x = x + rng.uniform(0, 0.1) * np.sin(2 * np.pi * rng.uniform(0.05, 0.5) * t + rng.uniform(0, 6.3))
        s = rng.integers(0, 101)
        return torch.from_numpy(x[:, s:s + 900].astype(np.float32)), y


def folds_arg(s):
    a, b = s.split("-") if "-" in s else (s, s)
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"), ap.add_argument("out")
    ap.add_argument("--folds", default="1-8"), ap.add_argument("--exclude", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40), ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    X, meta, folds = load(a.data)
    keep = [f for f in folds_arg(a.folds) if f != a.exclude]
    idx = np.where(np.isin(folds, keep))[0]
    print(f"training on folds {keep}: {len(idx)} records", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = LeadNet(len(CLASSES)).to(dev)
    loader = DataLoader(Train(X, idx, len(idx) * 2), batch_size=256, num_workers=6, persistent_workers=True,
                        worker_init_fn=lambda w: np.random.seed(a.seed * 100 + w))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.epochs * len(loader), pct_start=0.15)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.02)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(x)
                loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / n:.4f} acc {correct / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state": model.state_dict(), "classes": CLASSES, "folds": keep}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
