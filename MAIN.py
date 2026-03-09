import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

from services import check_liveness, extract_text, validate_document, verify_faces

app = FastAPI(title="Practical ID Verification Pipeline")

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


@app.post("/verify-user/")
async def verify_user(
    id_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...),
):
    id_path = UPLOAD_FOLDER / id_image.filename
    selfie_path = UPLOAD_FOLDER / selfie_image.filename

    try:
        with open(id_path, "wb") as buffer:
            shutil.copyfileobj(id_image.file, buffer)

        with open(selfie_path, "wb") as buffer:
            shutil.copyfileobj(selfie_image.file, buffer)
    except Exception as exc:
        return {
            "status": "Rejected",
            "step": "upload",
            "message": f"Failed to save uploaded files: {exc}",
        }

    doc_result = validate_document(str(id_path), template_dir="templates")

    liveness_result = check_liveness(
        stub_mode=os.getenv("LIVENESS_STUB", "1") == "1",
        environment=os.getenv("RUN_ENV", "local"),
    )

    ocr_result = extract_text(str(id_path))

    face_result = verify_faces(str(id_path), str(selfie_path))

    final_status = "Verified"
    failure_steps = []

    if not doc_result.get("valid", False):
        final_status = "Rejected"
        failure_steps.append("document_validation")

    if not liveness_result.get("live", False):
        final_status = "Rejected"
        failure_steps.append("liveness")

    if not face_result.get("verified", False):
        final_status = "Rejected"
        failure_steps.append("face_match")

    return {
        "status": final_status,
        "failure_steps": failure_steps,
        "document_validation": doc_result,
        "liveness": liveness_result,
        "ocr": ocr_result,
        "face_match": face_result,
    }
