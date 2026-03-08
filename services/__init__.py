from .document_validation import validate_document
from .face_match import verify_faces
from .liveness import check_liveness
from .ocr_service import extract_text

__all__ = [
    "validate_document",
    "verify_faces",
    "check_liveness",
    "extract_text",
]
