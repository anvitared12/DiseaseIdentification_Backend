import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf


def load_tflite_model(model_path: str, class_names_path: str):
    model_file = Path(model_path)
    print(f"[model] Looking for TFLite model at: {model_file.resolve()}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file.resolve()}")

    interpreter = tf.lite.Interpreter(model_path=str(model_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']   # e.g. [1, H, W, 3]
    img_size = (int(input_shape[1]), int(input_shape[2]))
    print(f"[model] Loaded TFLite model. Input shape: {input_shape}, img_size={img_size}")
    print(f"[model] Output shape: {output_details[0]['shape']}")

    names_file = Path(class_names_path)
    if names_file.exists():
        class_names = [
            line.strip()
            for line in names_file.read_text().splitlines()
            if line.strip()
        ]
        print(f"[model] Loaded {len(class_names)} class names")
    else:
        output_units = output_details[0]['shape'][-1]
        class_names = [f"plant_{i}" for i in range(output_units)]
        print(f"[model] Created {output_units} default class names")

    return interpreter, input_details, output_details, class_names, img_size


def preprocess(image: Image.Image, img_size: tuple) -> np.ndarray:
    img = image.resize(img_size)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict_disease(
    interpreter,
    input_details: list,
    output_details: list,
    class_names: list,
    image: Image.Image,
    img_size: tuple,
) -> dict:
    arr = preprocess(image, img_size)

    # TFLite expects the exact dtype the model was quantized with
    expected_dtype = input_details[0]['dtype']
    arr = arr.astype(expected_dtype)

    try:
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
    except Exception as e:
        print(f"[model] Prediction error: {e}")
        return {"disease_name": "unknown", "confidence": 0.0}

    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    disease_name = (
        class_names[top_idx] if top_idx < len(class_names) else f"class_{top_idx}"
    )
    print(f"[model] Predicted: {disease_name} ({confidence:.4f})")
    return {"disease_name": disease_name, "confidence": confidence}
