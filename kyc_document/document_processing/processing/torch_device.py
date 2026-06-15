"""Нормализация device для PyTorch (KYC_DEVICE=gpu → cuda)."""

from __future__ import annotations

import torch


def torch_device(device: str) -> torch.device:
    d = (device or "cpu").strip().lower()
    if d == "gpu":
        d = "cuda"
    if d.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(d)
