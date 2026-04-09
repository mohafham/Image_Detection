from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import torch, timm, io, base64
import numpy as np
from PIL import Image
from torchvision import transforms
from gradcam import generate_gradcam
from explain import generate_explanation
import cv2
import threading
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI Image Detector API",
    description="Detect AI-generated images using EfficientNet-B3 with Grad-CAM visualization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Class 0 = AI-GENERATED (FAKE folder in CIFAKE), Class 1 = REAL
LABELS = ["AI-GENERATED", "REAL"]

# Load model once at startup
import os
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.pth")

model = None
model_load_error = None
model_lock = threading.Lock()


def ensure_model_loaded():
    global model, model_load_error
    if model is not None:
        return model

    with model_lock:
        if model is not None:
            return model

        try:
            loaded_model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
            if os.path.exists(MODEL_PATH):
                loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                print(f"Model loaded successfully from {MODEL_PATH}")
            else:
                print(f"WARNING: Model file not found at {MODEL_PATH}")
                print("Using untrained model weights; predictions will be unreliable.")

            loaded_model.eval().to(DEVICE)
            model = loaded_model
            model_load_error = None
        except Exception as exc:
            model_load_error = str(exc)
            raise

    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    """Root endpoint - redirects to API documentation"""
    return {
        "message": "AI Image Detector API",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict (POST)"
        },
        "frontend": "http://localhost:8501"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        active_model = ensure_model_loaded()
    except Exception:
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {model_load_error}")

    # Load image
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    # Inference
    tensor = transform(img_resized).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = active_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()
        confidence = probs[pred].item()

    label = LABELS[pred]

    # Grad-CAM
    heatmap = generate_gradcam(active_model, tensor, img_np, pred)
    _, buffer = cv2.imencode(".png", cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    heatmap_b64 = base64.b64encode(buffer).decode("utf-8")

    # Gemini Explanation
    explanation = generate_explanation(label, confidence, "highlighted regions")

    return JSONResponse({
        "label": label,
        "confidence": round(confidence * 100, 2),
        "heatmap_base64": heatmap_b64,
        "explanation": explanation
    })

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path_exists": os.path.exists(MODEL_PATH)
    }
