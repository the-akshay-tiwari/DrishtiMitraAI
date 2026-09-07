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


def generate_gradcam(target_model: torch.nn.Module, image_tensor: torch.Tensor, target_class: int) -> str:
    h1 = None
    h2 = None
    try:
        target_layer = target_model.features[-1]
        features: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def forward_hook(module, input, output):
            features.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)

        input_tensor = image_tensor.clone().detach().requires_grad_(True)
        target_model.zero_grad()

        with torch.enable_grad():
            logits = target_model(input_tensor)
            score = logits[0, target_class]
            score.backward()

        if not features or not gradients:
            return ""

        feat = features[0]
        grad = gradients[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feat).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        cam_np = (cam.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
        h, w = cam_np.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
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
        print(f"Warning: GradCAM generation skipped due to error: {exc}")
        return ""
    finally:
        if h1 is not None:
            h1.remove()
        if h2 is not None:
            h2.remove()
        target_model.zero_grad()


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

    try:
        image_tensor = transform(rgb_image).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
        grade = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference computation error: {error}") from error

    try:
        heatmap_b64 = generate_gradcam(model, image_tensor, grade)
    except Exception:
        heatmap_b64 = ""

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



