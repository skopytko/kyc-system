import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import random
import argparse
import copy

import numpy as np

from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont, ImageChops
import cv2
from trdg.generators import GeneratorFromStrings
import pandas as pd


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_hex(color_hex: str) -> str:
    c = color_hex.strip()
    if not c.startswith("#"):
        c = f"#{c}"
    if len(c) == 4:  # #RGB -> #RRGGBB
        r, g, b = c[1], c[2], c[3]
        c = f"#{r}{r}{g}{g}{b}{b}"
    return c


def hex_to_rgb(color_hex: str) -> Tuple[int, int, int]:
    c = normalize_hex(color_hex).lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def generate_text_overlay(
    text: str,
    font_size: int,
    color_hex: str,
    trdg_args: Dict[str, Any],
    font_path: Optional[str],
) -> Image.Image:
    # TRDG cannot render on transparent directly; render on white then key out white
    # Build only supported args for the Python API
    args: Dict[str, Any] = {
        "skewing_angle": 0,
        "random_skew": False,
        "fit": True,
        "margins": (0, 0, 0, 0),
        "size": font_size,
        # Render pure black text so we can derive a clean alpha mask
        "text_color": "#000000",
        "background_type": int(trdg_args.get("background", 1)),
        "blur": int(trdg_args.get("blur", 0)),
        "random_blur": bool(trdg_args.get("random_blur", False)),
        "character_spacing": int(trdg_args.get("character_spacing", 0)),
        "space_width": float(trdg_args.get("space_width", 1.0)),
    }
    if font_path:
        args["fonts"] = [font_path]

    try:
        generator = GeneratorFromStrings([text], **args)
        img, _ = next(iter(generator))
    except (TypeError, AttributeError, Exception) as e:
        # Fallback for older/newer TRDG versions or compatibility issues: render via PIL directly
        # Measure text size
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        tmp = Image.new("L", (1, 1), color=255)
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), text, font=font)
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        img = Image.new("RGB", (w, h), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((-bbox[0], -bbox[1]), text, font=font, fill=(0, 0, 0))

    # Build alpha from inverted luminance (white background -> 0 alpha, black text -> 255)
    gray = img.convert("L")
    alpha = ImageOps.invert(gray)
    alpha = ImageOps.autocontrast(alpha)

    # Optional: thicken glyphs for bold appearance
    bold_radius = int(trdg_args.get("bold_radius", 0))
    bold_strength = float(trdg_args.get("bold_strength", 1.0))
    if bold_radius > 0:
        size = bold_radius * 2 + 1  # must be odd
        alpha_orig = alpha
        alpha_dilated = alpha.filter(ImageFilter.MaxFilter(size))
        s = max(0.0, min(1.0, bold_strength))
        alpha = Image.blend(alpha_orig, alpha_dilated, s)

    rgb = hex_to_rgb(color_hex)
    overlay = Image.new("RGBA", img.size, (rgb[0], rgb[1], rgb[2], 0))
    overlay.putalpha(alpha)
    return overlay


def composite_text(
    base: Image.Image,
    overlay: Image.Image,
    position: Tuple[int, int],
    max_width: Optional[int],
    angle: int,
) -> Image.Image:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))

    draw_img = overlay
    if max_width and overlay.width > max_width:
        scale = max_width / overlay.width
        new_size = (int(overlay.width * scale), int(overlay.height * scale))
        draw_img = overlay.resize(new_size, Image.LANCZOS)

    if angle:
        draw_img = draw_img.rotate(angle, expand=True, resample=Image.BICUBIC)

    x, y = position
    layer.paste(draw_img, (x, y), draw_img)
    return Image.alpha_composite(base.convert("RGBA"), layer)


def _sanitize_mrz_name(value: str) -> str:
    """Sanitize personal names for MRZ-like line: keep A-Z0-9, replace others with '<'."""
    if not isinstance(value, str):
        return ""
    s = value.upper()
    out_chars: list[str] = []
    for ch in s:
        if "A" <= ch <= "Z" or "0" <= ch <= "9":
            out_chars.append(ch)
        else:
            out_chars.append("<")
    # Collapse multiple '<' to single '<' to keep it tidy
    collapsed: list[str] = []
    for ch in out_chars:
        if ch == "<" and collapsed and collapsed[-1] == "<":
            continue
        collapsed.append(ch)
    return "".join(collapsed)


def _date_to_yymmdd(date_text: str) -> str:
    """Convert date like 'DD MM YYYY' to YYMMDD. Falls back to digits-only best effort."""
    if not isinstance(date_text, str):
        return "000000"
    parts = [p for p in date_text.replace(".", " ").replace("/", " ").split() if p]
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        # Heuristic: detect which token is year (length 4), the rest day & month
        if len(parts[2]) == 4:  # DD MM YYYY
            dd, mm, yyyy = parts[0], parts[1], parts[2]
        elif len(parts[0]) == 4:  # YYYY MM DD
            yyyy, mm, dd = parts[0], parts[1], parts[2]
        else:
            # Assume DD MM YY(YY)
            dd, mm, yyyy = parts[0], parts[1], parts[2]
            if len(yyyy) == 2:
                yyyy = f"20{yyyy}" if int(yyyy) < 50 else f"19{yyyy}"
        yy = int(yyyy) % 100
        return f"{yy:02d}{int(mm):02d}{int(dd):02d}"
    # Fallback: extract digits and try interpret
    digits = "".join(ch for ch in date_text if ch.isdigit())
    if len(digits) >= 8:
        yyyy = int(digits[0:4])
        mm = int(digits[4:6])
        dd = int(digits[6:8])
        return f"{yyyy%100:02d}{mm:02d}{dd:02d}"
    return "000000"


def _measure_text_width(
    text: str,
    font_size: int,
    color_hex: str,
    trdg_args: Dict[str, Any],
    font_path: Optional[str],
) -> int:
    overlay = generate_text_overlay(text, font_size, color_hex, trdg_args, font_path)
    return int(overlay.width)


def _fit_with_filler_to_width(
    base_text: str,
    filler_char: str,
    target_width_px: int,
    font_size: int,
    color_hex: str,
    trdg_args: Dict[str, Any],
    font_path: Optional[str],
) -> tuple[str, int]:
    """Append filler_char until width would exceed target. Returns (text, filler_count)."""
    text = base_text
    # Quick guard to avoid infinite loops with tiny target
    if target_width_px <= 0:
        return text, 0
    # Incrementally add filler as long as it fits
    filler_count = 0
    while True:
        candidate = text + filler_char
        w = _measure_text_width(candidate, font_size, color_hex, trdg_args, font_path)
        if w <= target_width_px:
            text = candidate
            filler_count += 1
            continue
        break
    return text, filler_count


def _list_image_files_recursive(root: Path) -> list[Path]:
    supported = {".png", ".jpg", ".jpeg", ".webp"}
    files: list[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in supported:
            files.append(p)
    return files


def _remove_white_background_rgba(img: Image.Image, tolerance: int = 15, soften: float = 1.5) -> Image.Image:
    # Make sure RGBA
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3].astype(np.uint8)
    # Chebyshev distance from white (255,255,255)
    dist = np.max(255 - rgb, axis=2)
    # Foreground where distance > tolerance (i.e., not near white)
    fg = dist > int(max(0, min(255, tolerance)))
    mask = np.where(fg, 255, 0).astype(np.uint8)

    mask_img = Image.fromarray(mask, mode="L")
    if soften and soften > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(soften)))

    # Combine with original alpha (keep transparent if it was already transparent)
    final_alpha = Image.fromarray(np.minimum(alpha, np.array(mask_img, dtype=np.uint8)), mode="L")
    out = rgba.copy()
    out.putalpha(final_alpha)
    return out


def _apply_otsu_alpha(
    img: Image.Image,
    soften: float = 1.0,
    open_kernel: int = 3,
    dilate_iter: int = 0,
) -> Image.Image:
    rgb = img.convert("RGB")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    # Invert so strokes (dark) become bright for THRESH_BINARY
    inv = 255 - gray
    _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    k = max(1, int(open_kernel))
    if k % 2 == 0:
        k += 1
    if k >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if int(dilate_iter) > 0:
        kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel_d, iterations=int(dilate_iter))
    if soften and soften > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), float(soften))
    rgba = img.convert("RGBA")
    alpha_existing = np.array(rgba.split()[3], dtype=np.uint8)
    final_alpha = np.minimum(alpha_existing, mask.astype(np.uint8))
    out = rgba.copy()
    out.putalpha(Image.fromarray(final_alpha, mode="L"))
    return out


def _apply_yellow_tint(img: Image.Image, tint_color_conf: Any = "255,250,230", tint_strength: float = 0.40) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    # Parse tint color
    tint_color = (255, 250, 230)
    try:
        if isinstance(tint_color_conf, str) and "," in tint_color_conf:
            tr, tg, tb = [int(x.strip()) for x in tint_color_conf.split(",")]
            tint_color = (
                max(0, min(255, tr)),
                max(0, min(255, tg)),
                max(0, min(255, tb)),
            )
        elif isinstance(tint_color_conf, (list, tuple)) and len(tint_color_conf) == 3:
            tr, tg, tb = [int(x) for x in tint_color_conf]
            tint_color = (
                max(0, min(255, tr)),
                max(0, min(255, tg)),
                max(0, min(255, tb)),
            )
    except Exception:
        tint_color = (255, 250, 230)

    try:
        s = max(0.0, min(1.0, float(tint_strength)))
    except Exception:
        s = 0.40

    r, g, b, a = img.split()
    rgb = Image.merge("RGB", (r, g, b))
    tint_layer = Image.new("RGB", rgb.size, tint_color)
    blended = Image.blend(rgb, tint_layer, alpha=s)
    out = Image.merge("RGBA", (*blended.split(), a))
    return out


def _apply_field_alpha(img: Image.Image, field: Dict[str, Any]) -> Image.Image:
    """Если в field указан alpha < 1, уменьшает непрозрачность изображения."""
    alpha_val = field.get("alpha", 1.0)
    if alpha_val is None or alpha_val >= 1.0:
        return img
    try:
        a = max(0.0, min(1.0, float(alpha_val)))
    except (TypeError, ValueError):
        return img
    rgba = img.convert("RGBA")
    r, g, b, old_a = rgba.split()
    new_a = old_a.point(lambda x: int(round(x * a)))
    out = Image.merge("RGBA", (r, g, b, new_a))
    return out


def _apply_color_multiply(img: Image.Image, color_conf: Any = "20,32,96", strength: float = 0.85) -> Image.Image:
    """Apply a strong color multiply tone while preserving alpha.
    Default color is dark blue; strength 0.85 gives a strong effect.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    # Parse color
    color = (20, 32, 96)
    try:
        if isinstance(color_conf, str) and "," in color_conf:
            cr, cg, cb = [int(x.strip()) for x in color_conf.split(",")]
            color = (
                max(0, min(255, cr)),
                max(0, min(255, cg)),
                max(0, min(255, cb)),
            )
        elif isinstance(color_conf, (list, tuple)) and len(color_conf) == 3:
            cr, cg, cb = [int(x) for x in color_conf]
            color = (
                max(0, min(255, cr)),
                max(0, min(255, cg)),
                max(0, min(255, cb)),
            )
    except Exception:
        color = (20, 32, 96)

    try:
        s = max(0.0, min(1.0, float(strength)))
    except Exception:
        s = 0.85

    r, g, b, a = img.split()
    rgb = Image.merge("RGB", (r, g, b))
    color_layer = Image.new("RGB", rgb.size, color)
    multiplied = ImageChops.multiply(rgb, color_layer)
    # Blend original with multiplied to control intensity
    toned = Image.blend(rgb, multiplied, alpha=s)
    out = Image.merge("RGBA", (*toned.split(), a))
    return out


def _resolve_font_path(root: Path, configured_path: Optional[str]) -> Optional[str]:
    """Разрешает путь к шрифту. Если указан OCRBPro — пробует fallback: ocrb10.ttf, ocrb.ttf (с Git)."""
    candidates: list[Path] = []
    if configured_path:
        candidate = (root.parent / configured_path).resolve()
        candidates.append(candidate)
        parent_dir = candidate.parent
        if parent_dir.exists():
            target_name_lower = candidate.name.lower()
            for entry in parent_dir.iterdir():
                if entry.name.lower() == target_name_lower:
                    candidates.append(entry.resolve())
                    break
            stem_lower = candidate.stem.lower()
            if stem_lower == "ocrbpro" or not any(c.exists() for c in candidates):
                # OCRBPro коммерческий, в репо нет — пробуем шрифты с Git
                for name in ("ocrb10.ttf", "ocrb.ttf"):
                    fallback = parent_dir / name
                    if fallback.exists():
                        candidates.append(fallback)
                        break
            if stem_lower == "calibri":
                # Calibri — ищем в docs_generator/fonts и стандартных путях (macOS / Windows)
                for p in (
                    parent_dir / "Calibri.ttf",
                    Path("/Library/Fonts/Microsoft/Calibri.ttf"),
                    Path(os.path.expandvars("${WINDIR}/Fonts/calibri.ttf")) if os.getenv("WINDIR") else Path("/none"),
                ):
                    if p.exists():
                        candidates.append(p.resolve())
                        break
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def render_once(root: Path, cfg: Dict[str, Any]) -> tuple[Path, Path]:
    template_path = Path(cfg["template_image"]) if os.path.isabs(cfg["template_image"]) else (root.parent / cfg["template_image"]).resolve()
    output_path = Path(cfg["output_image"]) if os.path.isabs(cfg["output_image"]) else (root.parent / cfg["output_image"]).resolve()

    default_font_path = _resolve_font_path(root, cfg.get("default_font_path"))

    trdg_defaults: Dict[str, Any] = cfg.get("default_trdg", {})

    os.makedirs(output_path.parent, exist_ok=True)

    base = Image.open(str(template_path)).convert("RGBA")
    overlay_canvas = Image.new("RGBA", base.size, (0, 0, 0, 0))

    # 1) Place image fields (e.g., signature, photo, mini_photo) if configured
    fields = cfg.get("fields", {})
    image_fields = [
        (fn, f) for fn, f in fields.items()
        if isinstance(f, dict) and ("image_dir" in f or f.get("same_as")) and ("position" in f or "bbox" in f)
    ]
    # Process "photo" before "mini_photo" so same_as can use the chosen image
    def _image_order(item: tuple) -> tuple:
        name = item[0]
        return (0 if name == "photo" else 1, name)
    image_fields.sort(key=_image_order)

    chosen_image_paths: Dict[str, Path] = {}
    for field_name, field in image_fields:
        image_dir_val = field.get("image_dir")
        if not image_dir_val and not field.get("same_as"):
            continue
        img_dir = (Path(image_dir_val) if os.path.isabs(image_dir_val) else (root.parent / image_dir_val).resolve()) if image_dir_val else None
        same_as = field.get("same_as")
        if same_as and same_as in chosen_image_paths:
            src_path = chosen_image_paths[same_as]
        else:
            if not img_dir:
                continue
            candidates = _list_image_files_recursive(img_dir)
            if not candidates:
                continue
            src_path = random.choice(candidates) if field.get("random", True) else candidates[0]
        chosen_image_paths[field_name] = src_path
        try:
            img = Image.open(str(src_path)).convert("RGBA")
            # Optional background removal (default True). For photos set remove_bg=false in config.
            if bool(field.get("remove_bg", True)):
                # Remove background using binarization (preferred for signatures), else near-white keying
                if bool(field.get("binarize", True)):
                    img = _apply_otsu_alpha(
                        img,
                        soften=float(field.get("soften", 1.0)),
                        open_kernel=int(field.get("open_kernel", 3)),
                        dilate_iter=int(field.get("dilate_iter", 0)),
                    )
                else:
                    img = _remove_white_background_rgba(
                        img,
                        tolerance=int(field.get("tolerance", 15)),
                        soften=float(field.get("soften", 1.5)),
                    )

            # Apply optional rotation BEFORE fitting/cropping
            angle = int(field.get("angle", 0))
            if angle:
                img = img.rotate(angle, expand=True, resample=Image.BICUBIC)

            # Decide if this is a signature-like field
            is_signature_like = ("signature" in str(field_name).lower()) or bool(field.get("is_signature", False))

            # If bbox is provided, fit into bbox; else use position + optional max dims
            if isinstance(field.get("bbox"), dict):
                bbox = field.get("bbox", {})
                x1 = int(bbox.get("x1", 0))
                y1 = int(bbox.get("y1", 0))
                x2 = int(bbox.get("x2", x1))
                y2 = int(bbox.get("y2", y1))
                target_w = max(1, x2 - x1)
                target_h = max(1, y2 - y1)
                fit_mode = str(field.get("fit", "contain")).lower()
                scale_w = target_w / img.width
                scale_h = target_h / img.height
                if fit_mode == "cover":
                    scale = max(scale_w, scale_h)
                    new_size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
                    img_rs = img.resize(new_size, Image.LANCZOS)
                    if is_signature_like:
                        img_rs = _apply_color_multiply(img_rs, field.get("signature_color", "20,32,96"), float(field.get("signature_strength", 0.85)))
                    elif bool(field.get("tint", True)):
                        img_rs = _apply_yellow_tint(img_rs, field.get("tint_color", "255,250,230"), float(field.get("tint_strength", 0.40)))
                    left = max(0, (img_rs.width - target_w) // 2)
                    top = max(0, (img_rs.height - target_h) // 2)
                    img_crop = img_rs.crop((left, top, left + target_w, top + target_h))
                    img_crop = _apply_field_alpha(img_crop, field)
                    overlay_canvas.paste(img_crop, (x1, y1), img_crop)
                else:  # contain
                    scale = min(scale_w, scale_h)
                    new_size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
                    img_rs = img.resize(new_size, Image.LANCZOS)
                    if is_signature_like:
                        img_rs = _apply_color_multiply(img_rs, field.get("signature_color", "20,32,96"), float(field.get("signature_strength", 0.85)))
                    elif bool(field.get("tint", True)):
                        img_rs = _apply_yellow_tint(img_rs, field.get("tint_color", "255,250,230"), float(field.get("tint_strength", 0.40)))
                    off_x = x1 + (target_w - img_rs.width) // 2
                    off_y = y1 + (target_h - img_rs.height) // 2
                    img_rs = _apply_field_alpha(img_rs, field)
                    overlay_canvas.paste(img_rs, (off_x, off_y), img_rs)
            else:
                # Optional scaling by max dimensions
                max_w = int(field.get("max_width", 0)) or None
                max_h = int(field.get("max_height", 0)) or None
                if max_w or max_h:
                    scale_w = (max_w / img.width) if (max_w and img.width > 0) else None
                    scale_h = (max_h / img.height) if (max_h and img.height > 0) else None
                    scales = [s for s in [scale_w, scale_h] if s]
                    if scales:
                        scale = min(scales)
                        new_size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
                        img = img.resize(new_size, Image.LANCZOS)
                if is_signature_like:
                    img = _apply_color_multiply(img, field.get("signature_color", "20,32,96"), float(field.get("signature_strength", 0.85)))
                elif bool(field.get("tint", True)):
                    img = _apply_yellow_tint(img, field.get("tint_color", "255,250,230"), float(field.get("tint_strength", 0.40)))

                x = int(field["position"]["x"])
                y = int(field["position"]["y"])
                img = _apply_field_alpha(img, field)
                overlay_canvas.paste(img, (x, y), img)
        except Exception as e:
            print(f"Warning: failed to place image for field '{field_name}' from {src_path}: {e}")

    # 2) Render text fields
    for field_name, field in cfg["fields"].items():
        if not isinstance(field, dict) or "text" not in field:
            continue
        text = field["text"]
        if str(field_name) == "authority_code":
            digits = "".join(c for c in str(text) if c.isdigit())
            text = (digits[-3:] if len(digits) >= 3 else digits.zfill(3))[:3]
        pos = (int(field["position"]["x"]), int(field["position"]["y"]))
        max_width = int(field.get("max_width", 0)) or None
        font_size = int(field.get("font_size", 32))
        angle = int(field.get("angle", 0))
        color_hex = field.get("color", "#000000")

        overlay = generate_text_overlay(
            text=text,
            font_size=font_size,
            color_hex=color_hex,
            trdg_args=trdg_defaults,
            font_path=default_font_path,
        )

        overlay_canvas = composite_text(overlay_canvas, overlay, pos, max_width, angle)

    # 3) Render MRZ (МЧЗ) lines
    try:
        fields = cfg.get("fields", {})
        val = lambda k, d="": str(fields.get(k, {}).get("text", d)) if isinstance(fields.get(k), dict) else d

        type_code = (val("type", "P") or "P").strip().upper()
        code_iss = (val("code_of_issuing", "BLR") or "BLR").strip().upper()
        surname = _sanitize_mrz_name(val("surname", ""))
        names = _sanitize_mrz_name(val("names", ""))
        passport_no = (val("passport_no", "") or "").strip().upper()
        dob_yymmdd = _date_to_yymmdd(val("date_of_birth", ""))
        doe_yymmdd = _date_to_yymmdd(val("date_of_expiry", ""))
        sex = (val("sex", "<") or "<").strip().upper()[:1]
        identification_no = (val("identification_no", "") or "").strip().upper()

        # MRZ line 1: type<'code_of_issuing''surname'<<'names'<<<... (fill with '<')
        mrz1_base = f"{type_code}<{code_iss}{surname}<<{names}"
        mrz_cfg = cfg.get("mrz", {})
        mrz_font_size = int(mrz_cfg.get("font_size", 32)) if mrz_cfg else 32
        mrz_color = "#2a2a2a"
        if mrz_cfg:
            mx = int(mrz_cfg.get("x", 90))
            my1 = int(mrz_cfg.get("y1", 1530))
            my2 = int(mrz_cfg.get("y2", 1590))
            mw = int(mrz_cfg.get("width", 1080))
            mrz_x1, mrz_y1, mrz_x2 = mx, my1, mx + mw
            mrz_x1_b, mrz_y2, mrz_x2_b = mx, my2, mx + mw
        else:
            mrz_x1, mrz_y1, mrz_x2 = 90, 1530, 1170
            mrz_x1_b, mrz_y2, mrz_x2_b = 90, 1590, 1170
        mrz_target_w = max(1, mrz_x2 - mrz_x1)
        mrz_font_path = _resolve_font_path(root, mrz_cfg.get("font_path")) if mrz_cfg else None  # None = старый встроенный шрифт для МЧЗ
        mrz1_fitted, _ = _fit_with_filler_to_width(
            base_text=mrz1_base,
            filler_char="<",
            target_width_px=mrz_target_w,
            font_size=mrz_font_size,
            color_hex=mrz_color,
            trdg_args=trdg_defaults,
            font_path=mrz_font_path,
        )
        mrz1_overlay = generate_text_overlay(
            text=mrz1_fitted,
            font_size=mrz_font_size,
            color_hex=mrz_color,
            trdg_args=trdg_defaults,
            font_path=mrz_font_path,
        )
        overlay_canvas = composite_text(
            overlay_canvas,
            mrz1_overlay,
            position=(mrz_x1, mrz_y1),
            max_width=mrz_target_w,
            angle=0,
        )

        # MRZ line 2: passport_no'5'code_of_issuing'date_of_birth(YYMMDD)'5'sex'date_of_expiry(YYMMDD)'5'identification_no + digits to fill
        mrz2_base = f"{passport_no}5{code_iss}{dob_yymmdd}5{sex}{doe_yymmdd}5{identification_no}"
        mrz_target_w2 = max(1, mrz_x2_b - mrz_x1_b)
        mrz2_fitted, _ = _fit_with_filler_to_width(
            base_text=mrz2_base,
            filler_char="0",
            target_width_px=mrz_target_w2,
            font_size=mrz_font_size,
            color_hex=mrz_color,
            trdg_args=trdg_defaults,
            font_path=mrz_font_path,
        )
        mrz2_overlay = generate_text_overlay(
            text=mrz2_fitted,
            font_size=mrz_font_size,
            color_hex=mrz_color,
            trdg_args=trdg_defaults,
            font_path=mrz_font_path,
        )
        overlay_canvas = composite_text(
            overlay_canvas,
            mrz2_overlay,
            position=(mrz_x1_b, mrz_y2),
            max_width=mrz_target_w2,
            angle=0,
        )
    except Exception as e:
        print(f"Warning: failed to render MRZ lines: {e}")

    filled = Image.alpha_composite(base, overlay_canvas)
    filled.save(str(output_path))
    print(f"Saved filled: {output_path}")
    return output_path


def main() -> None:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Render passport image(s) from config and optional CSV data overrides")
    parser.add_argument("--config", type=Path, default=root / "config.json", help="Path to config.json")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV with per-row field overrides")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write outputs when using CSV")
    parser.add_argument("--name-col", type=str, default=None, help="CSV column to use for output filename (without extension)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of rows to render from CSV")
    args = parser.parse_args()

    cfg = load_config(str(args.config))

    if not args.csv:
        render_once(root, cfg)
        return

    # CSV batch mode
    df = pd.read_csv(str(args.csv))
    if args.limit is not None and args.limit > 0:
        df = df.head(int(args.limit))

    # Determine base output directory
    base_out_path = Path(cfg["output_image"]) if os.path.isabs(cfg["output_image"]) else (root.parent / cfg["output_image"]).resolve()
    if args.output_dir:
        # If output_dir is specified, resolve it relative to docs_generator directory
        if os.path.isabs(str(args.output_dir)):
            out_dir = Path(args.output_dir)
        else:
            out_dir = (root.parent / args.output_dir).resolve()
    else:
        out_dir = base_out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        cfg_row = copy.deepcopy(cfg)
        # Override text fields from CSV when column matches field name
        fields = cfg_row.get("fields", {})
        for fname, fval in fields.items():
            if isinstance(fval, dict) and "text" in fval and fname in df.columns:
                val = row[fname]
                if pd.notna(val):
                    fval["text"] = str(val)
        # Output filename
        if args.name_col and args.name_col in df.columns and pd.notna(row[args.name_col]):
            name = str(row[args.name_col])
        else:
            name = f"passport_{idx+1:03d}"
        safe_name = name.replace(os.sep, "_").replace("/", "_").replace("\\", "_").strip()
        cfg_row["output_image"] = str(out_dir / f"{safe_name}.png")
        render_once(root, cfg_row)


if __name__ == "__main__":
    main()


