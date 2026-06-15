#!/usr/bin/env python3
"""
Прокачка изображений через Seedream 4.5 (BytePlus API).
Сохраняет в seedream_output/.

  export ARK_API_KEY="key"
  python scripts/seedream_process.py path/to/raw            # все в папке, 5 воркеров
  python scripts/seedream_process.py path/to/raw -j 5      # 5 воркеров
  python scripts/seedream_process.py path/to/img.jpg       # один файл
"""
import argparse
import csv
import io
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from openai import OpenAI
from openai import BadRequestError

BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
MODEL = "seedream-4-5-251128"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

BACKGROUND_VARIANTS = [
    "against the background of the old apartment interior",
    "against the background of a landfill",
    "against the background of a dog",
    "against the background of a cat",
    "the card lies against the background of wet asphalt",
    "against the background of a night metropolis",
    "against the background of a night campsite with a campfire",
    "against the background of a fireplace with a fire",
    "the card lies against the background of a car seat",
    "against the background of a bus steering wheel",
    "the card lies against the background of a dashboard car panels",
    "against the background of an airport",
    "against the background of the back of an airplane seat",
    "the card lies against the background of knees or hips",
    "the card lies against the background of bed linen",
    "a person holds the card in a very dark cave",
    "the card lies against the background of the ground",
    "the card lies against the background of a dirty kitchen area",
    "the wet card lies on the sand",
    "against the background of the sea",
    "against the background of a night lake without lighting",
    "the card lies on a stack of newspapers",
    "the card lies on its feet",
    "the card lies on a console",
    "against the background of a bathroom",
    "against the background of floor tiles",
    "the card lies on the background of a stone table",
    "the card lies on a wooden table",
    "the card lies on a velour upholstered chair",
    "the card lies on the hood of the car",
    "the card lies on the phone screen",
    "the card lies on the keyboard",
    "the card lies in a pencil case with pens",
    "the card lies among other paper documents",
    "the card is in a portmanteau",
    "the card lies on the washing machine",
    "the card is lying on the gas stove",
    "the card is lying on white meat",
    "the card is lying in bird feathers",
    "against the background of a border guard",
    "against the background of a small car with low lighting",
    "against the background of medals",
    "on a very dark night where nothing is visible on the card",
    "in the subway against the background of people",
    "the card is lying on a leather sofa",
    "the card is lying on shards glass",
]


# Временно отключить блок IMPORTANT в build_default_prompt()
INCLUDE_IMPORTANT = False


COLOR_GRADES = [
    "neutral natural color grade, balanced white point",
    "warm color grade (golden hour tone, slight amber cast)",
    "cool color grade (slight blue/teal cast, like cloudy daylight)",
    "indoor fluorescent tint (faint green-cyan cast)",
    "tungsten / incandescent indoor tint (warm orange cast)",
    "desaturated muted color grade with lifted blacks",
    "high-contrast cinematic grade with deep shadows",
    "slightly underexposed evening light, dimmer overall",
    "low-light dim ambient, document partially in shadow but still readable",
    "soft cloudy daylight, flat low-contrast lighting",
]

LIGHT_VARIANTS = [
    "soft diffuse ambient light from the top, very even on the document",
    "soft side window light from the left, gentle falloff to the right",
    "soft side window light from the right, gentle falloff to the left",
    "warm desk lamp from upper-left, mild shadow under the holding hand",
    "overhead ceiling light, soft directional shadow under the document edges",
    "low evening light, darker overall scene, document slightly dim but legible",
    "single warm light source, visible soft shadow cast by the hand and document",
    "natural daylight through a window, mild contrast",
]

EXTRA_REALISM_POOL = [
    "subtle aged paper look: faint yellowing at edges, very light wear marks, micro fiber texture",
    "very subtle light glare reflection on the laminate of the photo page, not covering text",
    "light fingerprint smudges on the laminate corners",
    "tiny dust specks on the paper surface",
    "barely-visible wear along the outer page edge, spine gutter only",
    "minor edge wear on the outer margin, no bent corners and no warping of text fields",
    "subtle vignette darkening at the frame corners (background only)",
    "soft shadow of the holding hand falling on part of the document",
    "very faint motion blur on the background, document stays perfectly sharp",
]


def load_csv_rows(csv_path: Optional[Path]) -> list:
    if not csv_path:
        return []
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_for_image(img: Path, rows: list) -> Optional[dict]:
    """Имя файла вида passport_0001.png → rows[0]."""
    if not rows:
        return None
    m = re.search(r"(\d+)(?!.*\d)", img.stem)
    if not m:
        return None
    idx = int(m.group(1)) - 1
    if 0 <= idx < len(rows):
        return rows[idx]
    return None


def build_text_fidelity_block(row: dict) -> str:
    """Точные значения полей паспорта → жёсткая инструкция модели."""
    lines: list[str] = []

    def add(label: str, value: Optional[str]) -> None:
        v = (value or "").strip()
        if v:
            lines.append(f'  - {label}: "{v}"')

    add("Issuing authority line 1", row.get("passport_issued"))
    add("Issuing authority line 2", row.get("passport_issued2"))
    add("Issuing authority line 3", row.get("passport_issued3"))
    add("Date of issue", row.get("date_of_issue"))
    add("Department code", row.get("department_code"))
    add("Surname (Cyrillic)", row.get("lastname"))
    add("First name (Cyrillic)", row.get("firstname"))
    add("Patronymic (Cyrillic)", row.get("middlename"))
    add("Sex", row.get("sex"))
    add("Date of birth", row.get("birth_date"))
    add("Place of birth", row.get("birth_place"))
    add("Series and number (vertical, red)", row.get("series_and_number"))

    if not lines:
        return ""
    body = "\n".join(lines)
    return (
        "TEXT GROUND TRUTH — these are the EXACT printed strings already visible on the "
        "input passport scan. Reproduce every one of them character-by-character. Do NOT "
        "translate, transliterate, paraphrase, abbreviate, replace, drop or add any "
        "character. Keep Cyrillic letters as Cyrillic, keep punctuation, keep dots in "
        "dates, keep the dash in the department code, keep the red vertical series/number "
        "exactly as printed:\n" + body
    )


def build_default_prompt(
    row: Optional[dict] = None,
    has_refs: bool = False,
    role_mode: str = "default",
) -> str:
    bg = random.choice(BACKGROUND_VARIANTS)
    grade = random.choice(COLOR_GRADES)
    light = random.choice(LIGHT_VARIANTS)
    extra_count = random.randint(2, 4)
    extras = random.sample(EXTRA_REALISM_POOL, k=extra_count)

    lines = [
        "Image-to-image edit. Use the provided scan as the document layer and change only the "
        "scene around it (background, lighting, hand, atmosphere).",
        "Highly realistic close-up photograph of an open Russian internal passport spread, "
        f"placed on this background: {bg}.",
        "CAMERA & ORIENTATION: the camera is strictly perpendicular (straight overhead, top-down, "
        "lens parallel to the open spread). The document must NOT be rotated in the frame — no "
        "diagonal placement, no 15–40° desk angle, no keystone/trapezoid perspective, no skew. "
        "Left and right page edges stay parallel to the image borders, like a proper flat KYC scan.",
        "BOOK SPINE (allowed bend only here): this is an open booklet. A gentle, natural curve "
        "along the center binding gutter / spine is OK — pages may dip slightly into the fold like "
        "a real passport book. Do NOT bend, curl, or lift the outer corners or page edges; do NOT "
        "warp the text fields. Only the spine area may show soft book curvature.",
        "The passport is held in a real human hand: thumb and a fingertip visible on the page "
        "edge or corner of the document, naturally gripping it. The hand may slightly overlap the "
        "outer edge or one corner of the passport but MUST NOT cover any printed text, photo, "
        "stamp, signature, or the spine area.",
        "Outer pages stay planar apart from the spine curve — no folding of corners, no rolling "
        "of edges, no perspective twist of the printed area.",
        "The passport must fill most of the frame (large, centered, minimal empty margin).",
        f"COLOR / GRADE: {grade}.",
        f"LIGHTING: {light}. Render realistic, well-defined shadows that match this light "
        "direction; shadows must look physically correct.",
        "REALISM DETAILS: the paper is real, slightly aged, used document — visible matte paper "
        "fibers, micro grain, faint surface microtexture, subtle laminate sheen on the photo page. "
        "It must look like a real photograph of a real passport, not a clean scan, not a render.",
        "Add the following extra realism details: " + "; ".join(extras) + ".",
        "PAPER & TEXTURE QUALITY: the paper surface must be sharp and in-focus across the entire "
        "document — visible fibers and microtexture, never blurry, never plasticky, never smoothed.",
        "TEXT FIDELITY (most important): every printed Cyrillic character, digit, line, stamp, "
        "field name and value must be a pixel-faithful copy of the source scan. Do not redraw, "
        "do not invent letters, do not change names, dates, numbers, codes, organization names, "
        "series/number — anything textual stays bit-for-bit identical to the input image.",
        "Do not blur, smear, soften, defocus, thicken, or stylize any text. Every glyph stays "
        "razor-sharp with crisp edges and clear ink-on-paper contrast.",
        "If you cannot keep a text region perfectly readable, leave the source pixels untouched "
        "in that region — never invent replacement glyphs.",
        "Output must be a sharp, high-resolution, photographic, KYC-grade image — no AI softness, "
        "no compression artifacts, no over-smoothing.",
    ]

    if has_refs:
        if role_mode == "swapped":
            lines.append(
                "MULTI-IMAGE INPUT (swapped roles): Image #1 is the PRIMARY output base "
                "(passport template v1). Image #2 is an alternate scan of the SAME person "
                "(passport2 template — different layout/typography). "
                "Edit ONLY Image #1: change background, lighting, hand, atmosphere. "
                "Keep Image #1's document layout, field positions and text rendering style. "
                "Image #2 is for cross-checking printed strings only — every name, date, "
                "number and Cyrillic character on the output must match Image #1 and Image #2 "
                "(they contain the same data). Do NOT redraw text in passport2's layout; "
                "do NOT merge or blend the two templates into one page."
            )
        elif role_mode == "text-from-ref":
            lines.append(
                "MULTI-IMAGE INPUT (text from reference): Image #1 is the PRIMARY scan "
                "(passport2 layout) — edit ONLY its scene/background/lighting/hand. "
                "Image #2 is the REFERENCE scan (passport v1) — treat ALL printed text "
                "(Cyrillic, digits, dates, codes, red series/number) as AUTHORITATIVE: "
                "reproduce every glyph exactly as in Image #2, but keep Image #1's page "
                "layout and composition. Do not invent characters; do not use passport2 "
                "text style if it disagrees with Image #2."
            )
        else:
            lines.append(
                "MULTI-IMAGE INPUT: you receive several scans of the SAME passport. "
                "Image #1 is the primary one — produce the output by editing only its "
                "scene/background/lighting/hand. The additional images are reference "
                "scans of THE SAME document with IDENTICAL printed text — cross-check "
                "every glyph, name, date and number against them and reproduce these "
                "characters EXACTLY. Do not blend the documents, do not invent characters "
                "that disagree with any reference."
            )

    if row:
        gt_block = build_text_fidelity_block(row)
        if gt_block:
            lines.append(gt_block)

    if INCLUDE_IMPORTANT:
        rotation_deg = random.randint(30, 40)
        spine_hidden_lines = random.randint(0, 5)
        important_pool = [
            f"Along the horizontal spine/binding gutter, partially cover {spine_hidden_lines} line(s) "
            "of text as if the document is slightly closed at the fold.",
            f"Rotate the document about {rotation_deg} degrees (not perfectly straight-on).",
            "Keep all printed text extremely sharp and legible (high detail on the document).",
            "Add strong, bright glare and reflections on the lower page of the document.",
            "Add strong, crisp, well-defined shadows on and around the document.",
            "The document must appear far from the camera (small in frame, strong sense of distance).",
            "Slightly overlap the document edges with fingers holding it (natural hand pose).",
            "Preserve all original printed content; do not change names, dates, or numbers.",
        ]
        pick_count = random.randint(0, 5)
        important_lines = random.sample(
            important_pool, k=min(pick_count, len(important_pool))
        )
        lines.extend(["", "IMPORTANT:"])
        lines.extend(f"- {item}" for item in important_lines)

    return "\n".join(lines)


def upload_to_litterbox(path: Path) -> str:
    """Загружает изображение на Litterbox (24ч), возвращает URL."""
    for attempt in range(5):
        try:
            with open(path, "rb") as f:
                r = requests.post(
                    "https://litterbox.catbox.moe/resources/internals/api.php",
                    data={"reqtype": "fileupload", "time": "24h"},
                    files={"fileToUpload": (path.name, f)},
                    timeout=90,
                )
            if r.status_code == 429:
                wait = 10 * (2 ** attempt)  # 10, 20, 40, 80 сек
                if attempt < 4:
                    time.sleep(wait)
                    continue
                r.raise_for_status()
            r.raise_for_status()
            url = r.text.strip()
            if not url.startswith("http"):
                raise RuntimeError(f"Litterbox: {url[:80]}")
            return url
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            if attempt < 4:
                time.sleep(10)
                continue
            raise


def generate(
    client: OpenAI,
    image_path: Path,
    prompt: str,
    size: str = "2K",
    square: bool = False,
    guidance_scale: float = 2.5,
    extra_image_paths: Optional[list] = None,
) -> list:
    """Отправляет image-to-image в Seedream, возвращает список URL.

    extra_image_paths — дополнительные изображения (тот же документ, другие сканы).
    Первое изображение в `image` — основное, остальные используются моделью как
    референсы и помогают точнее сохранить текст.
    """
    image_urls = [upload_to_litterbox(image_path)]
    for ref in extra_image_paths or []:
        try:
            image_urls.append(upload_to_litterbox(ref))
        except Exception:
            continue
    dim = 2048 if size == "2K" else 4096
    extra = {
        "image": image_urls,
        "image_size": size,
        "guidance_scale": guidance_scale,
        "watermark": False,
    }
    if square:
        extra["width"] = extra["height"] = dim
    else:
        from PIL import Image
        with Image.open(image_path) as im:
            w, h = im.size
        # Сохраняем пропорции, масштабируем к dim по длинной стороне
        scale = dim / max(w, h)
        extra["width"] = int(w * scale)
        extra["height"] = int(h * scale)
    resp = client.images.generate(
        model=MODEL,
        prompt=prompt,
        size=size,
        response_format="url",
        extra_body=extra,
    )
    return [d.url for d in resp.data]


def find_reference_pairs(img: Path, ref_dirs: list) -> list:
    """Ищет в `ref_dirs` файлы с тем же числовым суффиксом (_NNNN)."""
    m = re.search(r"(\d+)(?!.*\d)", img.stem)
    if not m:
        return []
    idx = m.group(1)
    pairs: list = []
    for ref_dir in ref_dirs:
        ref_dir = Path(ref_dir)
        if not ref_dir.is_dir():
            continue
        for p in ref_dir.iterdir():
            if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
                continue
            if p.resolve() == img.resolve():
                continue
            m2 = re.search(r"(\d+)(?!.*\d)", p.stem)
            if m2 and m2.group(1) == idx:
                pairs.append(p)
                break
    return pairs


def process_image(
    img: Path,
    out_dir: Path,
    prompt: Optional[str],
    size: str,
    square: bool,
    guidance_scale: float,
    api_key: str,
    print_lock: threading.Lock,
    rows: Optional[list] = None,
    ref_dirs: Optional[list] = None,
    role_mode: str = "default",
) -> bool:
    """Обрабатывает одно изображение. Возвращает True при успехе."""
    refs = find_reference_pairs(img, ref_dirs or [])
    if prompt is None:
        row = row_for_image(img, rows or [])
        prompt = build_default_prompt(row=row, has_refs=bool(refs), role_mode=role_mode)
    pattern = f"{img.stem}_seedream*{img.suffix}"
    existing = list(out_dir.glob(pattern))
    if existing:
        with print_lock:
            print(f"Пропускаю {img.name}: уже есть {len(existing)} файл(ов) seedream")
        return True

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    urls = None
    for attempt in range(3):
        try:
            urls = generate(
                client,
                img,
                prompt,
                size,
                square=square,
                guidance_scale=guidance_scale,
                extra_image_paths=refs,
            )
            if refs:
                with print_lock:
                    print(f"{img.name}: ref={[r.name for r in refs]}")
            break
        except BadRequestError as e:
            if "Timeout while downloading url" in str(e):
                if attempt < 2:
                    with print_lock:
                        print(f"{img.name}: BytePlus не скачал URL (попытка {attempt + 1}/3). Повтор...")
                    time.sleep(5)
                    continue
            raise
    if urls is None:
        return False

    for i, url in enumerate(urls):
        r = None
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                break
            except (requests.exceptions.ReadTimeout, requests.ConnectionError) as e:
                if attempt < 2:
                    with print_lock:
                        print(f"{img.name}: Ошибка скачивания (попытка {attempt + 1}/3). Повтор...")
                    time.sleep(3)
                    continue
                else:
                    with print_lock:
                        print(f"{img.name}: Не удалось скачать результат: {e}")
                    r = None
                    break
        if r is None:
            continue

        out = out_dir / f"{img.stem}_seedream{'_' + str(i) if len(urls) > 1 else ''}.png"
        data = r.content
        if square:
            from PIL import Image
            im = Image.open(io.BytesIO(data)).convert("RGB")
            w, h = im.size
            if w != h:
                sz = min(w, h)
                im = im.crop(((w - sz) // 2, (h - sz) // 2, (w + sz) // 2, (h + sz) // 2))
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=False, compress_level=1)
            data = buf.getvalue()
        out.write_bytes(data)
        with print_lock:
            print(f"Сохранено: {out.name}")

    return True


def main():
    p = argparse.ArgumentParser(description="Seedream 4.5 image-to-image")
    p.add_argument("input", nargs="?", type=Path, help="Файл или папка")
    p.add_argument("-o", "--output-dir", type=Path, help="Выход (по умолчанию: seedream_output/)")
    p.add_argument(
        "--prompt",
        default=None,
        help="Фиксированный промпт; без флага — на каждое изображение случайный фон из списка",
    )
    p.add_argument("--size", choices=["2K", "4K"], default="4K", help="Разрешение (по умолчанию 4K — лучше для читаемости текста)")
    p.add_argument(
        "--square",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Квадрат 1:1 (по умолчанию выключен — сохраняем пропорции для горизонтального паспорта)",
    )
    p.add_argument("-j", "--workers", type=int, default=5, help="Число параллельных воркеров (по умолчанию: 5)")
    p.add_argument(
        "--guidance",
        type=float,
        default=1.5,
        help="guidance_scale API (по умолчанию 1.5 — ближе к исходнику; выше => больше отклонений от текста)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("dataset/generator_docs/russia/personal_data/personal_data.csv"),
        help="CSV с данными (имя файла N → строка N). Используется для жёсткой инструкции модели.",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Не подмешивать ground-truth текст из CSV в промпт.",
    )
    p.add_argument(
        "--ref",
        action="append",
        default=[],
        help="Папка(и) с парными сканами того же документа (одинаковый _NNNN). "
             "Можно указать несколько раз: --ref dir1 --ref dir2.",
    )
    p.add_argument(
        "--role-mode",
        choices=("default", "swapped", "text-from-ref"),
        default="default",
        help="default: основной скан = выход; swapped (B): passport основной, "
             "passport2 референс; text-from-ref: passport2 основной, текст с passport.",
    )
    args = p.parse_args()

    api_key = os.environ.get("ARK_API_KEY") or "51a5645b-eb4c-439e-b0c8-b3025d312212"

    src = args.input
    if not src:
        raise SystemExit("Укажи файл или папку: python ... path/to/raw")

    src = src.resolve()
    if src.is_file() and src.suffix.lower() in IMAGE_EXTS:
        images = [src]
    elif src.is_dir():
        images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file())
        if not images:
            raise SystemExit(f"Нет изображений в {src}")
    else:
        raise SystemExit(f"Не найдено: {src}")

    out_dir = args.output_dir or Path(__file__).resolve().parent.parent / "seedream_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers)
    print_lock = threading.Lock()

    rows: list = []
    if not args.no_csv:
        csv_path = args.csv if args.csv.is_absolute() else Path.cwd() / args.csv
        if not csv_path.exists():
            csv_path = Path(__file__).resolve().parent.parent / args.csv
        rows = load_csv_rows(csv_path if csv_path.exists() else None)
        if rows:
            print(f"CSV ground-truth: {csv_path} ({len(rows)} строк)")
        else:
            print("CSV не подгружен — промпт без ground-truth")

    print(
        f"Обрабатываю {len(images)} файл(ов), {workers} воркер(ов), "
        f"guidance_scale={args.guidance}, square={args.square}, size={args.size}"
    )
    ex = ThreadPoolExecutor(max_workers=workers)
    ref_dirs = [Path(r) for r in args.ref]
    if ref_dirs:
        print(f"Reference scans: {[str(r) for r in ref_dirs]}")
    if args.role_mode != "default":
        print(f"Role mode: {args.role_mode}")

    futures = {
        ex.submit(
            process_image,
            img,
            out_dir,
            args.prompt,
            args.size,
            args.square,
            args.guidance,
            api_key,
            print_lock,
            rows,
            ref_dirs,
            args.role_mode,
        ): img
        for img in images
    }
    try:
        for fut in as_completed(futures):
            img = futures[fut]
            try:
                fut.result()
            except Exception as e:
                with print_lock:
                    print(f"Ошибка {img.name}: {e}")
    except KeyboardInterrupt:
        with print_lock:
            print("Ctrl+C: отменяю запланированные задачи, но даю завершиться текущим...")

        # cancel_futures=True отменяет только те future, которые еще не начали выполняться.
        # Уже запущенные задания продолжат выполняться, чтобы успели записать файлы.
        ex.shutdown(wait=True, cancel_futures=True)

        # Дополнительно выведем ошибки только для уже завершившихся заданий.
        for fut, img in futures.items():
            if fut.done() and not fut.cancelled():
                try:
                    fut.result()
                except Exception as e:
                    with print_lock:
                        print(f"Ошибка {img.name}: {e}")
        return
    finally:
        ex.shutdown(wait=True)


if __name__ == "__main__":
    main()
