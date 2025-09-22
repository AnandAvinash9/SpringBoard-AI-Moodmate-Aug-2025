import argparse, json, os
import cv2
import numpy as np
import tensorflow as tf
from src.utils.image import detect_and_crop_face

# Paths
MODEL_PATH = os.path.join("models", "fer_cnn.keras")
CLASS_JSON = os.path.join("models", "class_names.json")

def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_JSON, "r") as f:
        class_names = json.load(f)
    return model, class_names

def predict_emotion(image_path, model, class_names):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    crop = detect_and_crop_face(bgr)
    x = np.expand_dims(crop, axis=0)
    probs = model.predict(x, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    return class_names[pred_id], float(np.max(probs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to face image (jpg/png)")
    args = parser.parse_args()

    model, class_names = load_model()
    emotion, conf = predict_emotion(args.img, model, class_names)
    print(f"Predicted emotion: {emotion} (confidence {conf:.2f})")