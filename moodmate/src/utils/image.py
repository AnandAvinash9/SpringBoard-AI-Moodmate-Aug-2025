import cv2
import numpy as np

def detect_and_crop_face(bgr_image, fallback_square=48):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        crop = gray[y:y+h, x:x+w]
    else:
        # Fallback: use center square (face not found)
        h, w = gray.shape[:2]
        side = min(h, w)
        y0 = (h - side)//2
        x0 = (w - side)//2
        crop = gray[y0:y0+side, x0:x0+side]
    crop = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
    crop = crop.astype("float32") / 255.0
    crop = np.expand_dims(crop, axis=-1)
    return crop
