import argparse
import io
import os
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import cv2


def list_images(directory: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [p for p in sorted(directory.iterdir()) if p.suffix.lower() in exts]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def apply_vignette(pil_img: Image.Image, strength: float = 0.35) -> Image.Image:
    w, h = pil_img.size
    cx, cy = w / 2.0, h / 2.0
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    mask = 1.0 - strength * (dist / max_dist)
    mask = np.clip(mask, 0.5, 1.0)
    arr = np.array(pil_img).astype(np.float32)
    arr[..., :3] = arr[..., :3] * mask[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _low_freq_noise_map(w: int, h: int, scale: int = 64, octaves: int = 2) -> np.ndarray:
    sw = max(4, w // scale)
    sh = max(4, h // scale)
    base = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    for _ in range(max(1, octaves)):
        small = np.random.rand(sh, sw).astype(np.float32)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        base += up * amp
        amp *= 0.5
    base -= base.min()
    if base.max() > 1e-6:
        base /= base.max()
    return base


def uneven_illumination(pil_img: Image.Image, strength: float = 0.35) -> Image.Image:
    w, h = pil_img.size
    field = _low_freq_noise_map(w, h, scale=random.randint(48, 96), octaves=random.randint(1, 3))
    # Map to [1-s, 1+s] range but slightly biased
    s = max(0.05, min(0.6, strength))
    bias = random.uniform(-0.15, 0.15)
    illum = 1.0 + bias + (field - 0.5) * 2.0 * s
    illum = np.clip(illum, 0.6, 1.4)
    arr = np.array(pil_img).astype(np.float32)
    arr[..., :3] = arr[..., :3] * illum[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def scan_banding(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    y = np.arange(h, dtype=np.float32)
    period = random.uniform(40.0, 160.0)
    phase = random.uniform(0, 2 * np.pi)
    amp = random.uniform(0.02, 0.08)
    band = 1.0 + amp * np.sin(2 * np.pi * y / period + phase)
    # Small random noise
    band += np.random.normal(0.0, amp * 0.15, size=h).astype(np.float32)
    band = np.clip(band, 0.85, 1.2)
    arr = np.array(pil_img).astype(np.float32)
    arr[..., :3] = arr[..., :3] * band[:, None, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# Removed channel_misalignment (geometry-related)


def paper_texture(pil_img: Image.Image, strength: float = 0.12) -> Image.Image:
    w, h = pil_img.size
    # Create high-frequency texture by upscaling noise and applying high-pass
    base = _low_freq_noise_map(w, h, scale=random.randint(6, 12), octaves=1)
    blur = cv2.GaussianBlur(base, (0, 0), sigmaX=1.2)
    high = base - blur
    high = (high - high.min()) / (high.max() - high.min() + 1e-6) - 0.5
    tex = np.clip(1.0 + high * (strength * 2.0), 0.8, 1.2)
    arr = np.array(pil_img).astype(np.float32)
    arr[..., :3] = arr[..., :3] * tex[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def fold_lines(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    img = pil_img.convert("RGB")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    num = random.randint(0, 2)
    for _ in range(num):
        vertical = random.random() < 0.6
        if vertical:
            x = random.randint(int(0.1 * w), int(0.9 * w))
            a = random.randint(35, 95)
            draw.rectangle((x - 1, 0, x + 1, h), fill=(255, 255, 255, a))
        else:
            y = random.randint(int(0.1 * h), int(0.9 * h))
            a = random.randint(35, 95)
            draw.rectangle((0, y - 1, w, y + 1), fill=(255, 255, 255, a))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3.5)))
    out = Image.alpha_composite(img.convert("RGBA"), overlay)
    # Slight darkening around fold by subtracting blurred dark line
    dark_overlay = Image.new("L", (w, h), 0)
    d2 = ImageDraw.Draw(dark_overlay)
    if num > 0:
        # Add subtle dark counterpart
        for _ in range(num):
            if random.random() < 0.6:
                x = random.randint(int(0.1 * w), int(0.9 * w))
                d2.rectangle((x - 2, 0, x + 2, h), fill=random.randint(20, 50))
            else:
                y = random.randint(int(0.1 * h), int(0.9 * h))
                d2.rectangle((0, y - 2, w, y + 2), fill=random.randint(20, 50))
        dark_overlay = dark_overlay.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 4.0)))
        out_rgb = out.convert("RGB")
        shaded = Image.composite(Image.new("RGB", (w, h), (0, 0, 0)), out_rgb, dark_overlay)
        out = Image.blend(out_rgb, shaded, alpha=0.25)
    else:
        out = out.convert("RGB")
    return out


def gradient_color_cast(pil_img: Image.Image, max_shift: float = 0.06) -> Image.Image:
    w, h = pil_img.size
    field = _low_freq_noise_map(w, h, scale=random.randint(96, 160), octaves=1)
    # Each channel has its own slight field and bias
    shifts = [random.uniform(-max_shift, max_shift) for _ in range(3)]
    biases = [random.uniform(-0.03, 0.03) for _ in range(3)]
    arr = np.array(pil_img).astype(np.float32)
    for c in range(3):
        mult = 1.0 + biases[c] + (field - 0.5) * 2.0 * shifts[c]
        arr[..., c] = arr[..., c] * np.clip(mult, 0.85, 1.15)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_directional_shadow(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    overlay = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(overlay)
    # Random ellipse shadow parameters
    ell_w = int(w * random.uniform(0.6, 1.1))
    ell_h = int(h * random.uniform(0.2, 0.6))
    off_x = int(random.uniform(-0.2, 0.3) * w)
    off_y = int(random.uniform(-0.2, 0.5) * h)
    bbox = (off_x, off_y, off_x + ell_w, off_y + ell_h)
    opacity = int(random.uniform(30, 90))
    draw.ellipse(bbox, fill=opacity)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=random.uniform(40, 90)))
    rgb = pil_img.convert("RGB")
    shaded = Image.composite(Image.new("RGB", (w, h), (0, 0, 0)), rgb, overlay)
    # Blend lightly to avoid over-darkening
    out = Image.blend(rgb, shaded, alpha=random.uniform(0.15, 0.35))
    return out


def add_hard_shadow(pil_img: Image.Image) -> Image.Image:
    """Add a harder-edged cast shadow with slight softness for realism."""
    w, h = pil_img.size
    base = pil_img.convert("RGBA")
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    # Random polygonal shape to simulate a sharp shadow edge
    num_points = random.randint(4, 7)
    pts = []
    # Generate points around a random line crossing the image
    angle = random.uniform(-np.pi / 6, np.pi / 6)
    cx = random.uniform(0.2 * w, 0.8 * w)
    cy = random.uniform(0.2 * h, 0.8 * h)
    length = random.uniform(0.6 * max(w, h), 1.2 * max(w, h))
    thickness = random.uniform(0.08, 0.18) * max(w, h)
    # Direction vector
    dx = np.cos(angle)
    dy = np.sin(angle)
    # Build a rectangle-like polygon then jitter its points slightly
    p1 = (cx - dx * length / 2 - dy * thickness / 2, cy - dy * length / 2 + dx * thickness / 2)
    p2 = (cx + dx * length / 2 - dy * thickness / 2, cy + dy * length / 2 + dx * thickness / 2)
    p3 = (cx + dx * length / 2 + dy * thickness / 2, cy + dy * length / 2 - dx * thickness / 2)
    p4 = (cx - dx * length / 2 + dy * thickness / 2, cy - dy * length / 2 - dx * thickness / 2)
    poly = [p1, p2, p3, p4]
    jitter = int(0.02 * max(w, h))
    pts = [(int(x + random.randint(-jitter, jitter)), int(y + random.randint(-jitter, jitter))) for x, y in poly]
    draw.polygon(pts, fill=random.randint(120, 200))
    # Slight blur for softness, but much sharper than directional shadow
    mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 5.0)))
    shaded = Image.composite(Image.new("RGBA", (w, h), (0, 0, 0, 255)), base, mask)
    # Blend lightly to keep realism
    out = Image.blend(base, shaded, alpha=random.uniform(0.25, 0.45))
    return out.convert("RGB")


def add_gradient_shadow(pil_img: Image.Image) -> Image.Image:
    """Apply a large-scale gradient shadow across the image with slight softness.
    The gradient direction and strength are randomized for realism.
    """
    w, h = pil_img.size
    img = pil_img.convert("RGB")

    # Create normalized coordinate grid
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    # Random direction vector
    angle = random.uniform(0, 2 * np.pi)
    dx, dy = np.cos(angle), np.sin(angle)

    # Random center shift to avoid symmetry
    cx = random.uniform(0.2 * w, 0.8 * w)
    cy = random.uniform(0.2 * h, 0.8 * h)

    # Project coordinates onto direction vector and normalize to [0,1]
    proj = (dx * (x - cx) + dy * (y - cy))
    proj -= proj.min()
    denom = proj.max() - 0.0001
    proj = proj / denom

    # Shape the gradient and scale opacity
    gamma = random.uniform(1.2, 2.2)
    grad = np.clip(proj ** gamma, 0.0, 1.0)
    max_alpha = random.randint(80, 160)  # 0..255
    alpha = (grad * max_alpha).astype(np.uint8)

    # Slight blur for softness
    mask = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 5.0)))

    shadow_layer = Image.new("RGB", (w, h), (0, 0, 0))
    shaded = Image.composite(shadow_layer, img, mask)
    # Blend to control intensity
    out = Image.blend(img, shaded, alpha=random.uniform(0.25, 0.5))
    return out


def add_scuffs_and_speckles(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    out = pil_img.convert("RGBA")
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    # Random light scratches
    for _ in range(random.randint(6, 14)):
        x1 = random.randint(-int(0.1 * w), int(1.1 * w))
        y1 = random.randint(0, h)
        x2 = x1 + random.randint(int(0.05 * w), int(0.3 * w))
        y2 = y1 + random.randint(-int(0.03 * h), int(0.03 * h))
        a = random.randint(25, 70)
        thickness = random.randint(1, 2)
        draw.line((x1, y1, x2, y2), fill=(255, 255, 255, a), width=thickness)
    # Dark spots/smudges
    for _ in range(random.randint(30, 80)):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        r = random.randint(1, 3)
        a = random.randint(15, 60)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0, a))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
    out = Image.alpha_composite(out, layer)
    return out.convert("RGB")


# Removed perspective_jitter (geometry-related)


def blur_and_noise(pil_img: Image.Image) -> Image.Image:
    img = pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.2)))
    arr = np.array(img).astype(np.float32)
    sigma = random.uniform(3.0, 8.0)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def color_jitter(pil_img: Image.Image) -> Image.Image:
    img = pil_img
    # Slight random WB/contrast/saturation changes
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.95, 1.08))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.92, 1.10))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.90, 1.15))
    return img


def jpeg_artifacts(pil_img: Image.Image) -> Image.Image:
    # Re-encode to JPEG at random medium/low quality and back
    q = random.randint(35, 80)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=q, optimize=False, progressive=False)
    buf.seek(0)
    degraded = Image.open(buf).convert("RGB")
    # Optionally blend with original to control intensity
    return Image.blend(pil_img, degraded, alpha=random.uniform(0.5, 1.0))


def motion_blur(pil_img: Image.Image) -> Image.Image:
    """Apply linear motion blur with random length and angle."""
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    ksize = random.choice([3, 5, 7, 9, 11])
    angle = random.uniform(0, np.pi)
    # Build motion kernel
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    # Draw a line across the kernel center at specified angle
    center = ksize // 2
    dx = np.cos(angle)
    dy = np.sin(angle)
    # Sample points along the line
    for i in range(ksize):
        x = int(center + (i - center) * dx)
        y = int(center + (i - center) * dy)
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0
    s = kernel.sum()
    if s > 0:
        kernel /= s
    blurred = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return Image.fromarray(blurred)


def defocus_blur(pil_img: Image.Image) -> Image.Image:
    """Apply slight defocus via disk kernel."""
    img = np.array(pil_img.convert("RGB"))
    radius = random.choice([2, 3, 4, 5])
    k = radius * 2 + 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)).astype(np.float32)
    disk /= disk.sum() if disk.sum() > 0 else 1.0
    blurred = cv2.filter2D(img, -1, disk, borderType=cv2.BORDER_REPLICATE)
    return Image.fromarray(blurred)


def crop_margins(pil_img: Image.Image, left: int = 10, top: int = 10, right: int = 10, bottom: int = 20) -> Image.Image:
    w, h = pil_img.size
    l = max(0, int(left))
    t = max(0, int(top))
    r = max(0, int(right))
    b = max(0, int(bottom))
    # Ensure we leave at least 1 pixel
    x1 = min(l, max(0, w - 1))
    y1 = min(t, max(0, h - 1))
    x2 = max(x1 + 1, w - r)
    y2 = max(y1 + 1, h - b)
    return pil_img.crop((x1, y1, x2, y2))


def pipeline(pil_img: Image.Image, seed: Optional[int] = None, skip_crop: bool = False) -> Image.Image:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    img = pil_img.convert("RGB")

    # Geometry transforms removed

    # Uneven illumination field
    if random.random() < 0.95:
        img = uneven_illumination(img, strength=random.uniform(0.25, 0.50))

    # Lighting/shadow + vignette
    if random.random() < 0.85:
        img = add_directional_shadow(img)
    # Occasionally add a harder-edged cast shadow
    if random.random() < 0.40:
        img = add_hard_shadow(img)
    # Large-scale gradient shadow to break uniform tone
    if random.random() < 0.70:
        img = add_gradient_shadow(img)
    if random.random() < 0.90:
        img = apply_vignette(img, strength=random.uniform(0.25, 0.45))

    # Scan banding
    if random.random() < 0.80:
        img = scan_banding(img)

    # Surface wear
    if random.random() < 0.70:
        img = add_scuffs_and_speckles(img)

    # Paper texture
    if random.random() < 0.95:
        img = paper_texture(img, strength=random.uniform(0.08, 0.16))

    # Channel misalignment removed

    # Fold lines
    if random.random() < 0.55:
        img = fold_lines(img)

    # Blur, noise, color
    if random.random() < 0.90:
        img = blur_and_noise(img)
    # Additional blur variants
    if random.random() < 0.45:
        img = motion_blur(img)
    if random.random() < 0.35:
        img = defocus_blur(img)
    if random.random() < 0.95:
        img = color_jitter(img)

    # Gradient color cast
    if random.random() < 0.85:
        img = gradient_color_cast(img, max_shift=random.uniform(0.03, 0.08))

    # Compression artifacts at the end
    if random.random() < 0.95:
        img = jpeg_artifacts(img)

    # Final crop margins: L=30, T=25, R=20, B=70 (пропускается если skip_crop=True)
    if not skip_crop:
        img = crop_margins(img, left=30, top=25, right=20, bottom=70)
    return img


def process_directory(input_dir: Path, output_dir: Path, variants: int = 1, seed: Optional[int] = None) -> None:
    ensure_dir(output_dir)
    files = list_images(input_dir)
    if not files:
        print(f"No images found in {input_dir}")
        return
    for idx, src in enumerate(files, start=1):
        try:
            base = Image.open(str(src)).convert("RGB")
        except Exception as e:
            print(f"Skip {src.name}: {e}")
            continue
        for v in range(variants):
            s = None if seed is None else (seed + idx * 1000 + v)
            out_img = pipeline(base, seed=s)
            if variants == 1:
                out_name = src.name
            else:
                stem = src.stem
                out_name = f"{stem}_v{v+1}{src.suffix}"
            out_path = output_dir / out_name
            out_img.save(str(out_path))
            print(f"Saved: {out_path}")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Photorealistic augmentation for generated passport images")
    parser.add_argument("--input-dir", type=Path, default=root / "out", help="Directory with source images")
    parser.add_argument("--output-dir", type=Path, default=root / "out_aug", help="Where to save augmented images")
    parser.add_argument("--variants", type=int, default=1, help="How many variants per input image")
    parser.add_argument("--seed", type=int, default=None, help="Random seed base")
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir.is_absolute() else (root / args.input_dir)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (root / args.output_dir)
    process_directory(input_dir.resolve(), output_dir.resolve(), variants=max(1, int(args.variants)), seed=args.seed)


if __name__ == "__main__":
    main()


