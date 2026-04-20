from dotenv import load_dotenv
load_dotenv()

import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from model_loader import load_keras_model, predict_disease

app = FastAPI(title = "Disease Identification API")

MODEL_PATH = os.getenv("MODEL_PATH","Models/disease_identification_model.h5")

CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH","class_names.txt")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD","0.6"))

MODEL_PATH = r"D:\OJT sem4\Backend\Models\plantdisease_efficientnet_model.keras"
CLASS_NAMES_PATH = "class_names.txt"
CONFIDENCE_THRESHOLD = 0.40

model, class_names, img_size = load_keras_model(MODEL_PATH, CLASS_NAMES_PATH)

@app.get("/health")
def health():
    return {"status":"ok", "model_loaded":model is not None}

@app.post("/detect")
async def detect_disease(file: UploadFile = File(...)):
    if file.content_type not in("image/jpeg","image/png","image/webp"):
        raise HTTPException(status_code=400, detail="Unsupported image format")
    
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, details="Image too large (max 10 MB).")
    
    try:
        image=Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, details="Could not decode image.")
    
    keras_result = predict_disease(model, class_names, image, img_size)

    if keras_result["confidence"] >= CONFIDENCE_THRESHOLD:
        return JSONResponse({
            "disease_name": keras_result["disease_name"],
            "confidence": round(keras_result["confidence"], 4),
        })
    
    raise HTTPException(status_code=404, detail="Disease could not be identified with sufficient confidence.")