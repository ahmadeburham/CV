from pathlib import Path


def validate_document(image_path: str) -> dict:
    """Basic ID document quality validation based on readability and brightness."""
    path = Path(image_path)

    if not path.exists():
        return {"valid": False, "message": f"File not found: {path}"}

    try:
        import cv2
    except Exception as exc:
        return {"valid": False, "message": f"OpenCV unavailable: {exc}"}

    img = cv2.imread(str(path))
    if img is None:
        return {"valid": False, "message": f"Unreadable image: {path}"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    if brightness < 50:
        return {
            "valid": False,
            "message": "Document image is too dark for reliable verification.",
            "brightness": round(brightness, 2),
        }

    return {
        "valid": True,
        "message": "Document validation passed.",
        "brightness": round(brightness, 2),
    }
