"""Local FastAPI inference service for the trained DrishtiMitra research model."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision.models import efficientnet_b0
from torchvision.transforms import v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "drishtimitra_efficientnet_b0.pt"
DIST_DIR = PROJECT_ROOT / "dist"
ASSETS_DIR = DIST_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMAGE_BYTES = 12 * 1024 * 1024

app = FastAPI(title="DrishtiMitra experimental inference API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets top-level BEFORE route definitions so /assets/* is handled by StaticFiles
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

model: torch.nn.Module | None = None
class_names: list[str] = []
transform: v2.Compose | None = None


def pad_to_square(image: Image.Image, bg_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Pad image with black borders to make it 1:1 square without distorting aspect ratio."""
    width, height = image.size
    if width == height:
        return image
    max_dim = max(width, height)
    new_img = Image.new(image.mode, (max_dim, max_dim), bg_color)
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2
    new_img.paste(image, (paste_x, paste_y))
    return new_img


def validate_retina_image(image: Image.Image) -> tuple[bool, str]:
    """Validate whether an image is a valid, clear fundus retina image."""
    img_np = np.array(image.convert("RGB"))
    h, w, _ = img_np.shape

    if h < 64 or w < 64:
        return False, "Image resolution is too low. Please upload a clear fundus image."

    mean_intensity = float(np.mean(img_np))
    if mean_intensity < 10.0:
        return False, "Uploaded image is too dark or empty. Please upload an illuminated fundus retina image."
    if mean_intensity > 240.0:
        return False, "Uploaded image is overexposed. Please upload a clear fundus retina image."

    r_mean = float(np.mean(img_np[:, :, 0]))
    g_mean = float(np.mean(img_np[:, :, 1]))
    b_mean = float(np.mean(img_np[:, :, 2]))

    # In fundus retina images, Red channel dominates over Blue channel (R > B)
    if r_mean < b_mean * 1.05 and r_mean < 40.0:
        return False, "The uploaded image does not appear to be a retinal fundus image. Please upload a valid retina scan."

    # Center crop check (retina circular field)
    center_h_start, center_h_end = int(h * 0.2), int(h * 0.8)
    center_w_start, center_w_end = int(w * 0.2), int(w * 0.8)
    center_crop = img_np[center_h_start:center_h_end, center_w_start:center_w_end]

    center_r = float(np.mean(center_crop[:, :, 0]))
    center_b = float(np.mean(center_crop[:, :, 2]))

    if center_r < center_b * 1.08 and center_r < 35.0:
        return False, "The image does not match the color profile of a retinal fundus scan. Please upload a valid retina image."

    # Blur / Quality Check using discrete Laplacian variance
    gray = np.mean(img_np, axis=2).astype(np.float32)
    if gray.shape[0] > 10 and gray.shape[1] > 10:
        laplacian = (
            gray[2:, 1:-1] + gray[:-2, 1:-1] + gray[1:-1, 2:] + gray[1:-1, :-2] - 4 * gray[1:-1, 1:-1]
        )
        blur_variance = float(np.var(laplacian))
        if blur_variance < 5.0:
            return False, "Image is too blurry or out of focus to perform diabetic retinopathy screening. Please upload a clearer capture."

    return True, ""


def generate_cam(target_model: torch.nn.Module, feat: torch.Tensor, target_class: int) -> str:
    try:
        linear_layer: torch.nn.Linear | None = None
        for module in reversed(list(target_model.modules())):
            if isinstance(module, torch.nn.Linear):
                linear_layer = module
                break

        if linear_layer is None or linear_layer.weight is None:
            return ""

        class_weights = linear_layer.weight[target_class].view(1, -1, 1, 1)
        cam = (class_weights * feat).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        cam_np = (cam.squeeze().cpu().numpy() * 255).astype(np.uint8)
        h_dim, w_dim = cam_np.shape
        rgba = np.zeros((h_dim, w_dim, 4), dtype=np.uint8)
        rgba[:, :, 0] = np.clip(cam_np * 1.5, 0, 255).astype(np.uint8)
        rgba[:, :, 1] = np.clip((255 - np.abs(cam_np.astype(np.float32) - 128) * 2), 0, 255).astype(np.uint8)
        rgba[:, :, 2] = np.clip((255 - cam_np.astype(np.float32) * 1.5), 0, 255).astype(np.uint8)
        rgba[:, :, 3] = np.clip(cam_np.astype(np.float32) * 0.85, 0, 215).astype(np.uint8)

        overlay_img = Image.fromarray(rgba, mode="RGBA")
        buffer = io.BytesIO()
        overlay_img.save(buffer, format="PNG")
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
    except Exception as exc:
        print(f"Warning: CAM generation skipped due to error: {exc}")
        return ""


def load_model() -> None:
    global model, class_names, transform
    if not MODEL_PATH.is_file():
        return
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    normalization = checkpoint["normalization"]
    loaded_model = efficientnet_b0(weights=None)
    in_features = loaded_model.classifier[1].in_features
    loaded_model.classifier[1] = torch.nn.Sequential(torch.nn.Dropout(p=0.3), torch.nn.Linear(in_features, len(class_names)))
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    model = loaded_model.to(DEVICE).eval()
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=normalization["mean"], std=normalization["std"]),
    ])


@app.on_event("startup")
def startup() -> None:
    load_model()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ready" if model is not None else "model_not_trained",
        "device": str(DEVICE),
        "model_path": str(MODEL_PATH),
        "medical_disclaimer": "Experimental research software only; not for clinical diagnosis.",
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, object]:
    if model is None or transform is None:
        raise HTTPException(status_code=503, detail="No trained model found. Run backend/train.py first.")

    content_type = (image.content_type or "").lower()
    if content_type and not (content_type.startswith("image/") or content_type == "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP fundus image.")

    contents = await image.read(MAX_IMAGE_BYTES + 1)
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 12 MB limit.")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        with Image.open(io.BytesIO(contents)) as uploaded:
            rgb_image = uploaded.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from error

    # 1. Quality & Non-retina validation check
    is_valid_retina, error_reason = validate_retina_image(rgb_image)
    if not is_valid_retina:
        raise HTTPException(status_code=400, detail=error_reason)

    # 2. Aspect-ratio preserving square padding
    padded_image = pad_to_square(rgb_image)

    features: list[torch.Tensor] = []
    def forward_hook(module, input, output):
        features.append(output)

    target_layer = model.features[-1]
    hook_handle = target_layer.register_forward_hook(forward_hook)

    try:
        image_tensor = transform(padded_image).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
        grade = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    except Exception as error:
        print(f"Error during model prediction: {error}")
        raise HTTPException(status_code=500, detail=f"Inference computation error: {error}") from error
    finally:
        hook_handle.remove()

    heatmap_b64 = ""
    try:
        if features:
            heatmap_b64 = generate_cam(model, features[0], grade)
    except Exception as exc:
        print(f"Warning: Heatmap generation failed: {exc}")

    return {
        "grade": grade,
        "label": class_names[grade],
        "confidence": round(probabilities[grade], 4),
        "probabilities": {str(index): round(probability, 4) for index, probability in enumerate(probabilities)},
        "heatmap": heatmap_b64,
        "medical_disclaimer": "Experimental research output only. A qualified ophthalmologist must make the clinical decision.",
    }


# Mount compiled React assets and SPA fallback
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if full_path in {"health", "predict"}:
        raise HTTPException(status_code=404, detail="Not found")
    target_file = DIST_DIR / full_path
    if target_file.is_file():
        return FileResponse(target_file)
    index_file = DIST_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return {"message": "DrishtiMitra Inference API active. Run 'npm run build' to serve frontend."}



