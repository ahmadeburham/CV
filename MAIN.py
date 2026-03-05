from fastapi import FastAPI, File, UploadFile
import shutil
import os

from services.document_validation import validate_document
from services.liveness import check_liveness
from services.face_match import verify_faces
from services.ocr_service import extract_text

app = FastAPI(title="Real Estate CV Verification System")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/verify-user/")
async def verify_user(
    id_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...)
):

    id_path = f"{UPLOAD_FOLDER}/{id_image.filename}"
    selfie_path = f"{UPLOAD_FOLDER}/{selfie_image.filename}"

    with open(id_path, "wb") as buffer:
        shutil.copyfileobj(id_image.file, buffer)

    with open(selfie_path, "wb") as buffer:
        shutil.copyfileobj(selfie_image.file, buffer)

    # 1 Document Validation
    if not validate_document(id_path):
        return {"status": "Rejected", "reason": "Invalid Document"}

    # 2 Liveness Detection
    if not check_liveness():
        return {"status": "Rejected", "reason": "Liveness Failed"}

    # 3 Face Matching
    face_result = verify_faces(id_path, selfie_path)

    if not face_result["verified"]:
        return {"status": "Rejected", "reason": "Face Not Matching"}

    # 4 OCR Extraction
    text_data = extract_text(id_path)

    return {
        "status": "Verified",
        "similarity_score": face_result["similarity_score"],
        "extracted_text": text_data
    }