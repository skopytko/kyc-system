from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
import logging
import shutil
import uuid
import os
import numpy as np
import cv2
import json

from kyc_document.document_processing import Pipeline

from webapp.capture_validation import evaluate_capture

logger = logging.getLogger(__name__)


def sanitize_for_json(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Allowed upload formats (must match frontend hints & validation)
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ALLOWED_FORMATS_LABEL = "JPG, JPEG, PNG, BMP, TIF, TIFF"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="KYC System")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR.as_posix())


# Lazy pipeline initialization to avoid slow startup
pipeline_instance = None


def get_pipeline() -> Pipeline:
    global pipeline_instance
    if pipeline_instance is None:
        # Default to OpenVINO on CPU as in README; adjust via env if needed
        model_format = os.getenv("KYC_MODEL_FORMAT", os.getenv("RDOCR_MODEL_FORMAT", "ONNX"))
        device = os.getenv("KYC_DEVICE", os.getenv("RDOCR_DEVICE", "cpu"))
        pipeline_instance = Pipeline(model_format=model_format, device=device)
    return pipeline_instance


# ---------- Auth helpers ----------
def sha256_hex(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


AUTH_SALT = os.getenv("AUTH_SALT", "rdo-default-salt")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")  # set in env for production
AUTH_SECRET = os.getenv("AUTH_SECRET_KEY", "change-me-please")

app.add_middleware(SessionMiddleware, secret_key=AUTH_SECRET)


def get_password_verifier_hex() -> str:
    # verifier = sha256(salt + password)
    password = AUTH_PASSWORD or "admin123"
    return sha256_hex((AUTH_SALT + password).encode("utf-8"))


def require_auth(request: Request):
    if not request.session.get("auth"):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not request.session.get("auth"):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "allowed_formats_label": ALLOWED_FORMATS_LABEL,
            "allowed_suffixes": sorted(ALLOWED_SUFFIXES),
        },
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # Always refresh nonce on page load
    request.session["nonce"] = uuid.uuid4().hex
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/auth/nonce")
async def get_nonce(request: Request):
    # Issue a new nonce each time for one-time use
    nonce = uuid.uuid4().hex
    request.session["nonce"] = nonce
    return {"nonce": nonce, "salt": AUTH_SALT}


@app.post("/auth/login")
async def auth_login(request: Request, response_hash: str = Form(...)):
    nonce = request.session.get("nonce")
    if not nonce:
        return JSONResponse(status_code=400, content={"error": "Missing nonce"})
    verifier_hex = get_password_verifier_hex()
    expected = sha256_hex((verifier_hex + nonce).encode("utf-8"))
    if response_hash == expected:
        request.session["auth"] = True
        # Invalidate nonce after successful use
        request.session.pop("nonce", None)
        return {"ok": True}
    return JSONResponse(status_code=403, content={"error": "Invalid credentials"})


@app.post("/auth/logout")
async def auth_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.post("/api/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    _: None = Depends(require_auth),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Unsupported file type",
                "allowed_formats": ALLOWED_FORMATS_LABEL,
            },
        )

    # Save uploaded file to disk for consistent pipeline I/O
    uid = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{uid}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    pipeline = get_pipeline()

    # Проверка качества (блики, размытие и т.д.) выполняется всегда; отображение в UI — по чекбоксу на клиенте.
    result = pipeline(dest, check_quality=True)

    # Optionally save intermediate images into static for preview
    previews = {}
    try:
        inter_dir = STATIC_DIR / uid
        inter_dir.mkdir(parents=True, exist_ok=True)

        # original
        if result.meta_results.get("original_img") is not None:
            cv2.imwrite(
                (inter_dir / "original.jpg").as_posix(),
                cv2.cvtColor(result.meta_results["original_img"], cv2.COLOR_RGB2BGR),
            )
            previews["original"] = f"/static/{uid}/original.jpg"

        # rotated
        if getattr(result, "rotated_image", None) is not None:
            cv2.imwrite(
                (inter_dir / "rotated.jpg").as_posix(),
                cv2.cvtColor(result.rotated_image, cv2.COLOR_RGB2BGR),
            )
            previews["rotated"] = f"/static/{uid}/rotated.jpg"

        # doc detection borders
        if result.meta_results.get("DocDetector") and result.meta_results["DocDetector"].get("border_img") is not None:
            cv2.imwrite(
                (inter_dir / "doc_detection.jpg").as_posix(),
                cv2.cvtColor(result.meta_results["DocDetector"]["border_img"], cv2.COLOR_RGB2BGR),
            )
            previews["doc_detection"] = f"/static/{uid}/doc_detection.jpg"

        # perspective fixed
        if getattr(result, "img_with_fixed_perspective", None) is not None:
            cv2.imwrite(
                (inter_dir / "fixed_perspective.jpg").as_posix(),
                cv2.cvtColor(result.img_with_fixed_perspective, cv2.COLOR_RGB2BGR),
            )
            previews["fixed_perspective"] = f"/static/{uid}/fixed_perspective.jpg"

        # text fields overlay
        try:
            if result.text_fields is not None and result.img_with_fixed_perspective is not None:
                coords, _ = result.text_fields
                img_with_boxes = result.img_with_fixed_perspective.copy()
                for box in coords:
                    cv2.rectangle(
                        img_with_boxes,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 255, 0),
                        2,
                    )
                cv2.imwrite(
                    (inter_dir / "text_fields.jpg").as_posix(),
                    cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR),
                )
                previews["text_fields"] = f"/static/{uid}/text_fields.jpg"
        except Exception:
            pass

        # seals
        seal_meta = result.meta_results.get("PassportSealDetector") or {}
        for key in ("seals_img", "fixed_seal_img"):
            if seal_meta.get(key) is not None:
                target = inter_dir / f"{key}.jpg"
                cv2.imwrite(target.as_posix(), cv2.cvtColor(seal_meta[key], cv2.COLOR_RGB2BGR))
                previews[key] = f"/static/{uid}/{key}.jpg"

        # do not include word patches in previews per requirements
    except Exception:
        pass

    report = sanitize_for_json(getattr(result, "full_report", {}))
    meta_sanitized = sanitize_for_json(getattr(result, "meta_results", {}))
    capture = evaluate_capture(report, meta_sanitized)

    if not capture.get("ok"):
        logger.info(
            "capture_validation: upload_id=%s reasons=%s",
            uid,
            [r.get("code") for r in capture.get("reasons", [])],
        )

    return {
        "report": report,
        "previews": previews,
        "upload_id": uid,
        "filename": file.filename,
        "capture": capture,
    }


if __name__ == "__main__":
    import uvicorn

    _port = int(os.getenv("KYC_PORT", os.getenv("PORT", "8001")))
    _host = os.getenv("KYC_HOST", "127.0.0.1")
    uvicorn.run("webapp.main:app", host=_host, port=_port, reload=True)

