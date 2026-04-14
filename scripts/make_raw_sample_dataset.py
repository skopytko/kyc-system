#!/usr/bin/env python3
"""Собирает одну папку со случайной подвыборкой изображений из dataset/**/raw/.

Пример:
  python scripts/make_raw_sample_dataset.py --ratio 0.2 --out dataset/raw_sample_20pct

По умолчанию создаёт hardlink'и (быстро и без копирования данных).
Если hardlink невозможен (другая ФС) — можно выбрать --mode copy.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_raw_images(dataset_dir: Path):
    for p in dataset_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue
        parts = p.parts
        if "raw" not in parts or "labels" in parts:
            continue
        yield p


def safe_name(src: Path) -> str:
    # flatten path to avoid collisions: usa_alaska_raw_alaska_0001.png
    parts = [x for x in src.parts if x not in (".", "dataset")]
    return "__".join(parts)


def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            # fallback to copy if hardlink not possible
            shutil.copy2(src, dst)
            return
    raise ValueError(f"Unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset", help="Корень dataset/")
    ap.add_argument("--out", default="dataset/raw_sample_20pct", help="Куда положить выборку")
    ap.add_argument("--ratio", type=float, default=0.2, help="Доля выборки (0..1)")
    ap.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    ap.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    out_dir = Path(args.out).resolve()
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    paths = list(iter_raw_images(dataset_dir))
    if not paths:
        raise SystemExit(f"No raw images found under {dataset_dir}")

    rnd = random.Random(int(args.seed))
    rnd.shuffle(paths)

    n = max(1, int(round(len(paths) * float(args.ratio))))
    chosen = paths[:n]

    # write manifest
    (out_dir / "manifest.txt").write_text(
        "\n".join(str(p) for p in chosen) + "\n",
        encoding="utf-8",
    )

    # materialize files
    for p in chosen:
        name = safe_name(p.relative_to(dataset_dir))
        dst = images_dir / name
        link_or_copy(p, dst, args.mode)

    print(f"Raw images total: {len(paths)}")
    print(f"Chosen ({args.ratio:.3f}): {len(chosen)}")
    print(f"Output: {out_dir}")
    print(f"Images dir: {images_dir}")
    print(f"Mode: {args.mode}")


if __name__ == "__main__":
    main()

