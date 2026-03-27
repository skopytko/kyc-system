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
import io
import os
import random
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


def build_default_prompt() -> str:
    bg = random.choice(BACKGROUND_VARIANTS)
    return (
        "Edit this image: place the ID card against a new background in a realistic and natural setting. "
        f"Let the background be {bg}.\n"
        "MUST: The ID must be fully visible and held in the palm of your hand.\n"
        "The document should be slightly tilted away from you in all directions (natural perspective).\n"
        "Adjust the lighting of the document to match the surrounding environment; if it is nighttime, "
        "the document should be black and barely visible, and if it is daytime, there should be plenty "
        "of glare and shadows, and the document may also have bright, sharp glare, harsh shadows, small "
        "scratches on the plastic, or be in the shadows so that the text is barely visible.\n"
        "Extremely low image quality: like a photo from a phone from the early 2000s, very low resolution, "
        "a dull image, low contrast, very heavy grain and noise, digital noise, motion blur, "
        "out-of-focus areas, a dirty/smudged lens (spots, smudged areas), blown-out highlights and "
        "crushed shadows, uneven lighting, severe color distortions, distorted color tones, "
        "compression artifacts, and an overall poor appearance."
    )


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


def generate(client: OpenAI, image_path: Path, prompt: str, size: str = "2K", square: bool = False) -> list:
    """Отправляет image-to-image в Seedream, возвращает список URL."""
    image_url = upload_to_litterbox(image_path)
    dim = 2048 if size == "2K" else 4096
    extra = {
        "image": [image_url],
        "image_size": size,
        "guidance_scale": 7,
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


def process_image(
    img: Path,
    out_dir: Path,
    prompt: Optional[str],
    size: str,
    square: bool,
    api_key: str,
    print_lock: threading.Lock,
) -> bool:
    """Обрабатывает одно изображение. Возвращает True при успехе."""
    if prompt is None:
        prompt = build_default_prompt()
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
            urls = generate(client, img, prompt, size, square=square)
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

        out = out_dir / f"{img.stem}_seedream{'_' + str(i) if len(urls) > 1 else ''}{img.suffix}"
        data = r.content
        if square:
            from PIL import Image
            im = Image.open(io.BytesIO(data)).convert("RGB")
            w, h = im.size
            if w != h:
                sz = min(w, h)
                im = im.crop(((w - sz) // 2, (h - sz) // 2, (w + sz) // 2, (h + sz) // 2))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=95)
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
    p.add_argument("--size", choices=["2K", "4K"], default="2K", help="Разрешение")
    p.add_argument("--square", action="store_true", help="Квадрат 1:1 (по умолчанию — сохранять пропорции, паспорт целиком)")
    p.add_argument("-j", "--workers", type=int, default=5, help="Число параллельных воркеров (по умолчанию: 5)")
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

    print(f"Обрабатываю {len(images)} файл(ов), {workers} воркер(ов)")
    ex = ThreadPoolExecutor(max_workers=workers)
    futures = {
        ex.submit(
            process_image,
            img,
            out_dir,
            args.prompt,
            args.size,
            args.square,
            api_key,
            print_lock,
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
