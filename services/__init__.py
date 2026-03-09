from .document_validation import validate_document
from .face_match import extract_face, verify_faces
from .liveness import check_liveness
from .ocr_service import extract_text
from .template_loader import load_templates

__all__ = [
    "validate_document",
    "verify_faces",
    "extract_face",
    "check_liveness",
    "extract_text",
    "load_templates",
]
