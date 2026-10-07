"""PTB-XL superdiagnostic labels and the two diagnosis architectures used in the study."""
from __future__ import annotations

import ast
import csv

import numpy as np
import torch
from torch import nn

from .model import LeadNet

SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]


def superclass_labels(meta: list[dict], scp_statements_csv: str) -> np.ndarray:
    """Multi-hot superclass labels from all diagnostic SCP codes, as in PTB-XL's example code."""
    agg = {}
    for row in csv.DictReader(open(scp_statements_csv)):
        code = row[""] if "" in row else next(iter(row.values()))
        if row.get("diagnostic") in ("1", "1.0") and row.get("diagnostic_class"):
            agg[code] = row["diagnostic_class"]
    Y = np.zeros((len(meta), len(SUPER)), np.float32)
    for i, m in enumerate(meta):
        for code in ast.literal_eval(m["scp_codes"]):
            if code in agg:
                Y[i, SUPER.index(agg[code])] = 1.0
    return Y


class InceptionBlock(nn.Module):
    def __init__(self, cin: int, nf: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(cin, nf, 1, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(nf, nf, k, padding=k // 2, bias=False) for k in (39, 19, 9)])
        self.pool = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(cin, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(4 * nf)

    def forward(self, x):
        b = self.bottleneck(x)
        return torch.relu(self.bn(torch.cat([c(b) for c in self.convs] + [self.pool(x)], 1)))


class InceptionTime(nn.Module):
    """An InceptionTime-style network (Ismail Fawaz et al. 2020) with residual shortcuts every 3 blocks."""

    def __init__(self, n_classes: int, nf: int = 32, depth: int = 6):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        cin = 12
        for d in range(depth):
            self.blocks.append(InceptionBlock(cin, nf))
            if d % 3 == 2:
                self.shortcuts.append(nn.Sequential(nn.Conv1d(12 if d == 2 else 4 * nf, 4 * nf, 1, bias=False), nn.BatchNorm1d(4 * nf)))
            cin = 4 * nf
        self.head = nn.Linear(8 * nf, n_classes)

    def forward(self, x):
        res = x
        for d, block in enumerate(self.blocks):
            x = block(x)
            if d % 3 == 2:
                x = torch.relu(x + self.shortcuts[d // 3](res))
                res = x
        return self.head(torch.cat([x.mean(-1), x.amax(-1)], 1))


def build(arch: str) -> nn.Module:
    return LeadNet(len(SUPER)) if arch == "resnet" else InceptionTime(len(SUPER))
