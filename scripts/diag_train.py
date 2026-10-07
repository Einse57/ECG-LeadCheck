"""Train a PTB-XL superdiagnostic model. usage: diag_train.py DATA_DIR SCP_CSV OUT.pt --arch resnet|inception --seed S [--swap-aug]"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from leadcheck.data import finish, load  # noqa: E402
from leadcheck.diagnosis import SUPER, build, superclass_labels  # noqa: E402
from leadcheck.swaps import CLASSES, apply  # noqa: E402


class Train(Dataset):
    def __init__(self, X, Y, idx, swap_aug):
        self.X, self.Y, self.idx, self.aug = X, Y, idx, swap_aug

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        j = self.idx[i]
        kind = CLASSES[int(rng.integers(1, len(CLASSES)))] if self.aug and rng.random() < 0.5 else "none"
        x = finish(apply(self.X[j], kind, rng))
        x = x * rng.uniform(0.9, 1.1) + rng.normal(0, 0.01, x.shape)
        s = rng.integers(0, 101)
        return torch.from_numpy(x[:, s:s + 900].astype(np.float32)), torch.from_numpy(self.Y[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"), ap.add_argument("scp"), ap.add_argument("out")
    ap.add_argument("--arch", default="resnet"), ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40), ap.add_argument("--swap-aug", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    X, meta, folds = load(a.data)
    Y = superclass_labels(meta, a.scp)
    keep = Y.sum(1) > 0
    tr = np.where((folds <= 8) & keep)[0]
    va = np.where((folds == 9) & keep)[0]
    dev = "cuda"
    model = build(a.arch).to(dev)
    loader = DataLoader(Train(X, Y, tr, a.swap_aug), batch_size=128, shuffle=True, num_workers=6, persistent_workers=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.epochs * len(loader), pct_start=0.2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xva = torch.from_numpy(np.stack([finish(X[j]) for j in va]))
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                p = torch.cat([torch.sigmoid(model(Xva[s:s + 512].to(dev)).float()).cpu() for s in range(0, len(Xva), 512)]).numpy()
            auc = np.mean([roc_auc_score(Y[va, k], p[:, k]) for k in range(len(SUPER))])
            print(f"epoch {ep + 1}: fold-9 macro AUROC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state": model.state_dict(), "arch": a.arch, "swap_aug": a.swap_aug, "seed": a.seed}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
