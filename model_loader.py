import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf

def load_keras_model(model_path: str, class_names_path: str):
    model_file = Path(model_path)
    print(f"[model] Looking for model at: {model_file.resolve()}")

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file.resolve()}")
    model = tf.keras.models.load_model(str(model_file))

    input_shape = model.input_shape
    img_size = (input_shape[1],input_shape[2])
    print(f"[model] Loaded. Input shape: {input_shape}, img_size={img_size}")
    print(f"[model] Output classes: {model.output_shape[-1]}")

    names_file = Path(class_names_path)
    if names_file.exists():
        class_names = [line.strip() for line in names_file.read_text().splitlines() if line.strip()]
        print(f"[model] Loaded {len(class_names)} class names")
    else:
        output_units = model.output_shape[-1]
        class_names = [f"plant_{i}" for i in range(output_units)]
        print(f"[model] Created default class names: {output_units} generic labels")

    return model, class_names, img_size  


def preprocess(image: Image.Image, img_size: tuple) -> np.ndarray:
    img = image.resize(img_size)
    arr = np.array(img, dtype=np.float32)
    
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    
    return np.expand_dims(arr, axis=0)

def predict_disease(model, class_names: list, image: Image.Image, img_size: tuple) -> dict:
    arr = preprocess(image, img_size)
    try:
        preds = model.predict(arr, verbose=0)[0]
    except Exception as e:
        print(f"[model] Prediction error: {e}")
        return {"disease_name": "unknown", "confidence": 0.0}

    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    disease_name = class_names[top_idx] if top_idx < len(class_names) else f"class_{top_idx}"
    print(f"[model] Predicted: {disease_name} ({confidence:.4f})")
    return {"disease_name": disease_name, "confidence": confidence}