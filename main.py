from dotenv import load_dotenv
load_dotenv()

import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from model_loader import load_tflite_model, predict_disease

app = FastAPI(title="Disease Identification API")

MODEL_PATH = os.getenv("MODEL_PATH", "d_model.tflite")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "class_names.txt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))

interpreter, input_details, output_details, class_names, img_size = load_tflite_model(
    MODEL_PATH, CLASS_NAMES_PATH
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": interpreter is not None}


@app.post("/detect")
async def detect_disease(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Unsupported image format")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10 MB).")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    result = predict_disease(
        interpreter, input_details, output_details, class_names, image, img_size
    )

    if result["confidence"] >= CONFIDENCE_THRESHOLD:
        return JSONResponse({
            "disease_name": result["disease_name"],
            "confidence": round(result["confidence"], 4),
        })

    raise HTTPException(
        status_code=404,
        detail="Disease could not be identified with sufficient confidence.",
    )
