"""A small 1D ResNet over the 12 leads."""
from __future__ import annotations

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(cin, cout, 9, stride, 4, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, 9, 1, 4, bias=False), nn.BatchNorm1d(cout))
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class LeadNet(nn.Module):
    def __init__(self, n_classes: int, width: int = 64):
        super().__init__()
        w = [width, width * 2, width * 3, width * 4]
        self.stem = nn.Sequential(nn.Conv1d(12, w[0], 15, 2, 7, bias=False), nn.BatchNorm1d(w[0]), nn.ReLU(inplace=True))
        layers, cin = [], w[0]
        for i, c in enumerate(w):
            layers += [Block(cin, c, 1 if i == 0 else 2), Block(c, c, 1)]
            cin = c
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(2 * cin, n_classes)

    def forward(self, x):
        h = self.body(self.stem(x))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))
