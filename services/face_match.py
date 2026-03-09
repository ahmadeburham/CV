import tempfile
from pathlib import Path
from typing import Dict, Optional


def _load_face_cascade(cv2_module):
    return cv2_module.CascadeClassifier(cv2_module.data.haarcascades + "haarcascade_frontalface_default.xml")


def extract_face(image_path: str, output_path: Optional[str] = None, margin: float = 0.2) -> Dict:
    try:
        import cv2
    except Exception as exc:
        return {"success": False, "face_path": None, "message": f"OpenCV unavailable: {exc}"}

    path = Path(image_path)
    if not path.exists():
        return {"success": False, "face_path": None, "message": f"Image not found: {path}"}

    image = cv2.imread(str(path))
    if image is None:
        return {"success": False, "face_path": None, "message": f"Unreadable image: {path}"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = _load_face_cascade(cv2)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return {"success": False, "face_path": None, "message": "No face detected."}

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad_w, pad_h = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(image.shape[1], x + w + pad_w), min(image.shape[0], y + h + pad_h)
    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        return {"success": False, "face_path": None, "message": "Face crop failed."}

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(prefix="face_", suffix=".jpg", delete=False)
        output_path = tmp.name
        tmp.close()

    if not cv2.imwrite(output_path, cropped):
        return {"success": False, "face_path": None, "message": "Failed to save face crop."}

    return {
        "success": True,
        "face_path": output_path,
        "message": "Face extracted successfully.",
        "bbox": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
    }


def verify_faces(id_path: str, selfie_path: str, threshold: float = 0.68) -> Dict:
    id_file = Path(id_path)
    selfie_file = Path(selfie_path)

    if not id_file.exists():
        return {"verified": False, "distance": None, "threshold": threshold, "message": f"ID image not found: {id_file}"}
    if not selfie_file.exists():
        return {"verified": False, "distance": None, "threshold": threshold, "message": f"Selfie image not found: {selfie_file}"}

    id_face = extract_face(str(id_file))
    if not id_face.get("success"):
        return {"verified": False, "distance": None, "threshold": threshold, "message": f"ID portrait extraction failed: {id_face.get('message')}"}

    selfie_face = extract_face(str(selfie_file))
    if not selfie_face.get("success"):
        return {"verified": False, "distance": None, "threshold": threshold, "message": f"Selfie face detection failed: {selfie_face.get('message')}"}

    try:
        from deepface import DeepFace

        result = DeepFace.verify(
            img1_path=id_face["face_path"],
            img2_path=selfie_face["face_path"],
            model_name="ArcFace",
            distance_metric="cosine",
            enforce_detection=False,
            detector_backend="skip",
        )
        distance = float(result.get("distance", 1.0))
        verified = bool(result.get("verified", distance <= threshold))
        return {
            "verified": verified,
            "distance": round(distance, 4),
            "threshold": threshold,
            "message": "Face verification completed using cropped ID portrait and selfie.",
        }
    except Exception as exc:
        return {"verified": False, "distance": None, "threshold": threshold, "message": f"Face verification failed: {exc}"}
