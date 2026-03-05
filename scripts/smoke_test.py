from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MAIN import app
from services import document_validation, face_match, liveness, ocr_service


def main():
    assert app is not None
    assert callable(document_validation.validate_document)
    assert callable(face_match.verify_faces)
    assert callable(liveness.check_liveness)
    assert callable(ocr_service.extract_text)
    print("IMPORTS OK")


if __name__ == "__main__":
    main()
