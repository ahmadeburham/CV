import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

from services import check_liveness, extract_text, validate_document, verify_faces

app = FastAPI(title="Real Estate CV Verification System")

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


@app.post("/verify-user/")
async def verify_user(
    id_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...),
):
    id_path = UPLOAD_FOLDER / id_image.filename
    selfie_path = UPLOAD_FOLDER / selfie_image.filename

    with open(id_path, "wb") as buffer:
        shutil.copyfileobj(id_image.file, buffer)

    with open(selfie_path, "wb") as buffer:
        shutil.copyfileobj(selfie_image.file, buffer)

    doc_result = validate_document(str(id_path))
    if not doc_result.get("valid"):
        return {"status": "Rejected", "step": "document_validation", "detail": doc_result}

    liveness_result = check_liveness(stub_mode=os.getenv("LIVENESS_STUB", "1") == "1")
    if not liveness_result.get("live"):
        return {"status": "Rejected", "step": "liveness", "detail": liveness_result}

    face_result = verify_faces(str(id_path), str(selfie_path))
    if not face_result.get("verified"):
        return {"status": "Rejected", "step": "face_match", "detail": face_result}

    ocr_result = extract_text(str(id_path))

    return {
        "status": "Verified",
        "document_validation": doc_result,
        "liveness": liveness_result,
        "face_match": face_result,
        "ocr": ocr_result,
    }
