import os
import random
import io
import yaml
import torch
import torch.nn.functional as F
import ollama
from PIL import Image
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from torchvision import transforms

from models.mobilenet import get_model
from models.verification import load_verifier, verify_image

CONFIG_PATH = os.environ.get('CONFIG_PATH', 'configs/config.yaml')
MODEL_PATH = os.environ.get('MODEL_PATH', 'results/best_model.pth')
VERIFIER_PATH = os.environ.get('VERIFIER_PATH', 'results/verifier_model.pth')
DEFAULT_VERIFICATION_THRESHOLD = float(os.environ.get("VERIFICATION_THRESHOLD", 0.5))
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "data" / "train"

app = FastAPI(title="PeelSafe CNN Banana Leaf Disease Prediction API")

app.mount("/static", StaticFiles(directory=IMAGES_DIR), name="static")

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = None
model = None
verifier = None
transform = None
classes = ['Banana_Bract_Mosaic_Virus_Disease', 'Banana_Cordana_Disease', 'Banana_Healthy', 'Banana_Insect_Pest_Disease', 'Banana_Moko_Disease', 'Banana_Panama_Disease', 'Banana_Pestalotiopsis_Disease', 'Banana_Yellow_and_Black_Sigatoka_Disease']

@app.on_event("startup")
async def startup_event():
    """Load models and configuration on startup"""
    global config, model, verifier, transform, device
    
    print("Loading configuration...")
    config = load_config(CONFIG_PATH)
    
    print("Loading transforms...")
    transform = get_transform(config)
    
    print("Loading disease classification model...")
    model = load_model(config, MODEL_PATH, device)
    
    print("Loading verification model...")
    verifier = load_verifier(VERIFIER_PATH, device)
    
    print("API startup complete!")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def get_transform(cfg):
    return transforms.Compose([
        transforms.Resize(cfg['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg['augmentation']['normalize']['mean'],
            std=cfg['augmentation']['normalize']['std']
        )
    ])

def load_model(cfg, model_path: str, device: torch.device):
    m = get_model(cfg)
    m.load_state_dict(torch.load(model_path, map_location=device))
    m = m.to(device)
    m.eval()
    return m

def predict_pil_image(pil_image: Image.Image, verification_threshold: float = DEFAULT_VERIFICATION_THRESHOLD):
    """
    Synchronous function that returns the same shaped dict your terminal script printed.
    Designed to be run in a threadpool from FastAPI.
    """
    global model, verifier, transform, device, classes

    # Preprocess
    image_tensor = transform(pil_image).unsqueeze(0).to(device)  # shape (1,C,H,W)

    # Verification (your verify_image expects one-image tensor)
    is_banana_leaf, verification_confidence = verify_image(image_tensor[0], verifier, verification_threshold)
    # verification_confidence is assumed to be in 0..1. Apply threshold override.
    if (not is_banana_leaf) or (verification_confidence < verification_threshold):
        return {
            'is_banana_leaf': False,
            'verification_confidence': float(verification_confidence) * 100.0,
            'message': 'The image does not appear to contain a banana leaf.'
        }

    # Disease prediction
    with torch.no_grad():
        outputs = model(image_tensor)                     # shape (1, num_classes)
        probabilities = F.softmax(outputs, dim=1)[0]     # shape (num_classes,)
        predicted_idx = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted_idx].item())

    return {
        'is_banana_leaf': True,
        'verification_confidence': float(verification_confidence) * 100.0,
        'class': classes[predicted_idx],
        'confidence': confidence * 100.0,
        'probabilities': {
            cls: float(prob.item()) * 100.0
            for cls, prob in zip(classes, probabilities)
        }
    }

@app.get("/")
async def root():
    
    return {"message": "CNN Model API is running"}

@app.get("/health/")
async def health():
    return {"status": "ok", "framework": "PyTorch"}

@app.get("/classes")
async def get_classes():
    return {"classes": classes}

@app.post("/predict")
async def predict(file: UploadFile = File(...), verification_threshold: float = Query(DEFAULT_VERIFICATION_THRESHOLD, ge=0.0, le=1.0)):
    print(f"Received file: {file.filename}, content_type: {file.content_type}")
    
    # Check if models are loaded
    if model is None or verifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please wait for startup to complete.")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    print(f"File size: {len(contents)} bytes")
    
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        print(f"Image size: {pil_image.size}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Run inference in threadpool to avoid blocking event loop
    try:
        result = await run_in_threadpool(predict_pil_image, pil_image, verification_threshold)

        # Only call Ollama if a banana leaf is detected
        if result.get('is_banana_leaf') and 'class' in result:
            disease_name = result['class'].replace('_', ' ')
            prompt = f"Provide a concise but informative explanation about {disease_name} in banana plants, including its causes, symptoms, and prevention methods."

            try:
                ollama_response = ollama.chat(
                    model='symonvalencia/peelsafeV1:latest',  # replace with your pulled model’s name
                    messages=[
                        {"role": "system", "content": "You are an agricultural expert specializing in banana diseases."},
                        {"role": "user", "content": prompt}
                    ]
                )
                context = ollama_response['message']['content']
                result['ollama_context'] = context
            except Exception as e:
                result['ollama_context'] = f"Ollama context unavailable: {str(e)}"

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
@app.get("/disease-images")
async def get_disease_images(name: str = Query(..., description="Disease name")):
    disease_dir = IMAGES_DIR / name
    if not disease_dir.exists() or not disease_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"No sample images found for {name}")

    image_files = [
        f for f in os.listdir(disease_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    if not image_files:
        raise HTTPException(status_code=404, detail=f"No image files found for {name}")

    sample_count = min(4, len(image_files))
    sample_files = random.sample(image_files, sample_count)

    base_url = "http://localhost:8000"
    image_urls = [f"{base_url}/static/{name}/{f}" for f in sample_files]

    random.shuffle(image_urls)

    return {"images": image_urls}