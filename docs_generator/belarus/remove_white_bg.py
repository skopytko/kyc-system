from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_propagation


def iter_image_files(root: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _kmeans_rgb(samples: np.ndarray, k: int, max_iters: int = 12, seed: int = 0) -> np.ndarray:
    """Tiny k-means on RGB samples; returns centers as float32 (k' x 3)."""
    samples_f = samples.astype(np.float32)
    n = samples_f.shape[0]
    unique_colors = np.unique(samples_f, axis=0)
    k_eff = max(1, min(k, unique_colors.shape[0], n))
    if k_eff == 1:
        return unique_colors[:1]
    rng = np.random.default_rng(seed)
    centers = np.empty((k_eff, 3), dtype=np.float32)
    idx0 = int(rng.integers(0, n))
    centers[0] = samples_f[idx0]
    d2 = np.sum((samples_f - centers[0]) ** 2, axis=1)
    for i in range(1, k_eff):
        denom = float(d2.sum())
        if denom <= 1e-12:
            idx = int(rng.integers(0, n))
        else:
            probs = d2 / denom
            probs = probs / probs.sum()
            idx = int(rng.choice(n, p=probs))
        centers[i] = samples_f[idx]
        d2 = np.minimum(d2, np.sum((samples_f - centers[i]) ** 2, axis=1))
    for _ in range(max_iters):
        dists = np.sqrt(((samples_f[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for i in range(k_eff):
            group = samples_f[labels == i]
            if group.size:
                new_centers[i] = group.mean(axis=0)
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return centers


def remove_white_background(
    image_path: Path,
    output_path: Path,
    white_threshold: int = 245,
    soften_radius: float = 1.5,
    crop_to_content: bool = True,
    pad: int = 0,
    tolerance: int = 15,
    bg_color: tuple[int, int, int] | None = None,
    bg_border: int = 8,
    bg_k: int = 2,
    aspect_w: int | None = None,
    aspect_h: int | None = None,
    anchor: str = "bottom",
) -> None:
    image = Image.open(str(image_path)).convert("RGBA")
    rgba = np.array(image)

    # Work in signed space to safely subtract
    rgb = rgba[..., :3].astype(np.int16)
    alpha = rgba[..., 3].astype(np.int16)

    # Determine solid background color(s)
    if bg_color is None:
        h, w, _ = rgb.shape
        b = int(max(1, min(bg_border, max(1, min(h, w) // 8))))
        # Sample only the top edge (full width) and corner squares
        top = rgb[0:b, :, :]
        tl = rgb[0:b, 0:b, :]
        tr = rgb[0:b, w - b : w, :]
        bl = rgb[h - b : h, 0:b, :]
        br = rgb[h - b : h, w - b : w, :]
        samples = np.concatenate(
            (
                top.reshape(-1, 3),
                tl.reshape(-1, 3),
                tr.reshape(-1, 3),
                bl.reshape(-1, 3),
                br.reshape(-1, 3),
            ),
            axis=0,
        )
        if samples.size == 0:
            centers = np.array([[255, 255, 255]], dtype=np.float32)
        else:
            if int(bg_k) <= 1:
                centers = np.median(samples, axis=0, keepdims=True).astype(np.float32)
            else:
                centers = _kmeans_rgb(samples.astype(np.float32), int(bg_k))
    else:
        centers = np.array([bg_color], dtype=np.float32)

    tol = int(max(0, min(255, tolerance)))
    # Candidate background pixels: close to ANY background center
    rgb_f = rgb.astype(np.float32)
    diffs = np.abs(rgb_f[:, :, None, :] - centers[None, None, :, :])  # H W K 3
    dists = diffs.max(axis=3)  # H W K
    min_dist = dists.min(axis=2)  # H W
    candidates = min_dist <= tol

    # Propagate from image edges through candidate regions (removes interior background touching edges)
    seeds = np.zeros_like(candidates, dtype=bool)
    seeds[0, :] = candidates[0, :]
    seeds[-1, :] = candidates[-1, :]
    seeds[:, 0] = candidates[:, 0]
    seeds[:, -1] = candidates[:, -1]
    is_bg = binary_propagation(seeds, mask=candidates)

    new_alpha = np.where(is_bg, 0, 255).astype(np.uint8)
    mask = Image.fromarray(new_alpha, mode="L")

    if soften_radius and soften_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(soften_radius)))

    if image.mode == "RGBA":
        orig_alpha = Image.fromarray(alpha.astype(np.uint8), mode="L")
        # Combine by taking the minimum alpha to ensure background is removed
        combined = Image.eval(mask, lambda m: m)  # copy
        combined_np = np.minimum(np.array(orig_alpha, dtype=np.uint8), np.array(mask, dtype=np.uint8))
        final_alpha = Image.fromarray(combined_np, mode="L")
    else:
        final_alpha = mask

    out = image.copy()
    out.putalpha(final_alpha)

    if crop_to_content:
        bbox = final_alpha.getbbox()
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            if pad > 0:
                x0 = max(0, x0 - pad)
                y0 = max(0, y0 - pad)
                x1 = min(out.width, x1 + pad)
                y1 = min(out.height, y1 + pad)
            out = out.crop((x0, y0, x1, y1))

    # Adjust to target aspect ratio by changing height only.
    # Policy: always pad (increase height) at the TOP if needed; if cropping is needed, crop from the BOTTOM.
    if aspect_w and aspect_h and out.width > 0:
        target_h = max(1, int(round(out.width * (aspect_h / float(aspect_w)))))
        h = out.height
        w = out.width
        if target_h < h:
            # Need to crop: remove from bottom only
            crop_amt = h - target_h
            box = (0, 0, w, h - crop_amt)  # crop bottom
            out = out.crop(box)
        elif target_h > h:
            # Need to pad: add space at top only
            pad_amt = target_h - h
            if out.mode != "RGBA":
                out = out.convert("RGBA")
            # Create transparent canvas (preserve transparency)
            canvas = Image.new("RGBA", (w, target_h), (0, 0, 0, 0))
            y = pad_amt  # place original at bottom; new space at top
            canvas.paste(out, (0, y))
            out = canvas

    out.save(str(output_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove near-white backgrounds from images, producing transparent PNGs.",
    )
    default_input = Path(__file__).resolve().parent / "photo"
    default_output = default_input / "no-bg"
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Directory with source images (default: passport_generator/photo)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory to write outputs (default: <input-dir>/no-bg)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Filename suffix for outputs (before extension)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=245,
        help="RGB channel threshold for considering a pixel as white background (0-255)",
    )
    parser.add_argument(
        "--soften",
        type=float,
        default=1.5,
        help="Gaussian blur radius for mask edges to reduce halos (pixels)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable cropping to content (default is to crop)",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=0,
        help="Optional padding (in pixels) around cropped content",
    )
    # removed white background compositing; always keep transparency
    parser.add_argument(
        "--bg-color",
        type=str,
        default=None,
        help="Force solid background color as R,G,B (e.g. 250,250,250). If omitted, auto-detect from borders.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=15,
        help="Tolerance (0-255) for background color matching (Chebyshev distance)",
    )
    parser.add_argument(
        "--bg-border",
        type=int,
        default=8,
        help="Border width (px) used to sample background color when auto-detecting",
    )
    parser.add_argument(
        "--bg-k",
        type=int,
        default=2,
        help="Number of border color clusters (k-means) to model multi-tone backgrounds",
    )
    parser.add_argument(
        "--aspect",
        type=str,
        default=None,
        help="Target aspect ratio as W:H (e.g. 35:45). Adjusts height only.",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        choices=["top", "bottom", "center"],
        default="bottom",
        help="Which edge remains fixed when adjusting height (default: bottom)",
    )

    args = parser.parse_args(argv)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    suffix: str = args.suffix
    threshold: int = max(0, min(255, int(args.threshold)))
    soften: float = max(0.0, float(args.soften))

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input dir not found or not a directory: {input_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    # Parse aspect if provided
    aspect_w: int | None = None
    aspect_h: int | None = None
    if args.aspect:
        sep = ":" if ":" in args.aspect else ("x" if "x" in args.aspect else ("/" if "/" in args.aspect else None))
        if sep is None:
            print(f"Invalid --aspect format: {args.aspect}. Expected W:H, e.g. 35:45", file=sys.stderr)
            return 2
        try:
            aw, ah = args.aspect.split(sep)
            aspect_w = int(aw)
            aspect_h = int(ah)
            if aspect_w <= 0 or aspect_h <= 0:
                raise ValueError
        except Exception:
            print(f"Invalid --aspect values: {args.aspect}. Use two positive integers like 35:45.", file=sys.stderr)
            return 2

    # Parse bg-color if provided
    forced_bg: tuple[int, int, int] | None = None
    if args.bg_color:
        try:
            parts = [int(x.strip()) for x in str(args.bg_color).split(',')]
            if len(parts) != 3:
                raise ValueError
            forced_bg = tuple(max(0, min(255, v)) for v in parts)  # type: ignore[assignment]
        except Exception:
            print("Invalid --bg-color. Use format R,G,B with integers 0-255.", file=sys.stderr)
            return 2

    for src in iter_image_files(input_dir):
        dst_name = f"{src.stem}{suffix}.png"
        dst = output_dir / dst_name
        remove_white_background(
            src,
            dst,
            white_threshold=threshold,
            soften_radius=soften,
            crop_to_content=(not args.no_crop),
            pad=int(args.pad),
            tolerance=int(max(0, min(255, getattr(args, "tolerance", 15)))),
            bg_color=forced_bg,
            bg_border=int(max(1, getattr(args, "bg_border", 8))),
            bg_k=int(max(1, getattr(args, "bg_k", 2))),
            aspect_w=aspect_w,
            aspect_h=aspect_h,
            anchor=str(args.anchor),
        )
        processed += 1
        print(f"Saved: {dst}")

    if processed == 0:
        print("No images found to process.")
    else:
        print(f"Processed {processed} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


