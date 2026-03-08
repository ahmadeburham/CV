from pathlib import Path


def verify_faces(id_path: str, selfie_path: str) -> dict:
    id_file = Path(id_path)
    selfie_file = Path(selfie_path)

    if not id_file.exists():
        return {"verified": False, "message": f"ID image not found: {id_file}"}

    if not selfie_file.exists():
        return {"verified": False, "message": f"Selfie image not found: {selfie_file}"}

    try:
        from deepface import DeepFace
    except Exception as exc:
        return {"verified": False, "message": f"DeepFace unavailable: {exc}"}

    try:
        result = DeepFace.verify(
            img1_path=str(id_file),
            img2_path=str(selfie_file),
            model_name="ArcFace",
            distance_metric="cosine",
            enforce_detection=False,
        )
    except Exception as exc:  # DeepFace can fail on missing faces/model issues
        return {
            "verified": False,
            "message": f"Face verification failed: {exc}",
        }

    distance = float(result.get("distance", 1.0))
    similarity = max(0.0, 1 - distance)

    return {
        "verified": bool(result.get("verified", False)),
        "similarity_score": round(similarity, 3),
        "message": "Face verification completed.",
    }
