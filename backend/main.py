from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import torch, timm, io, base64
import numpy as np
from PIL import Image
from torchvision import transforms
from gradcam import generate_gradcam
from explain import generate_explanation
import cv2
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

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"WARNING: Model file not found at {MODEL_PATH}")
    print("Please train the model first or place best_model.pth in the models/ directory")
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
    model.eval().to(DEVICE)
else:
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    print(f"Model loaded successfully from {MODEL_PATH}")

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
    # Load image
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    # Inference
    tensor = transform(img_resized).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()
        confidence = probs[pred].item()

    label = LABELS[pred]

    # Grad-CAM
    heatmap = generate_gradcam(model, tensor, img_np, pred)
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
    return {"status": "ok"}
