from pathlib import Path

from services import check_liveness, extract_text, validate_document, verify_faces


ID_IMAGE_PATH = "/kaggle/working/CV/uploads/id.jpg"
SELFIE_IMAGE_PATH = "/kaggle/working/CV/uploads/selfie.jpg"


def _print_step(name: str, result: dict):
    print(f"\n[{name}]")
    for key, value in result.items():
        print(f"- {key}: {value}")


def main():
    id_path = Path(ID_IMAGE_PATH)
    selfie_path = Path(SELFIE_IMAGE_PATH)

    print("=== Kaggle CV Verification Test ===")
    print(f"ID image: {id_path}")
    print(f"Selfie image: {selfie_path}")

    doc_result = validate_document(str(id_path))
    _print_step("Document Validation", doc_result)
    if not doc_result.get("valid"):
        print("\nPipeline stopped: document validation failed.")
        return

    liveness_result = check_liveness(stub_mode=True)
    _print_step("Liveness", liveness_result)
    if not liveness_result.get("live"):
        print("\nPipeline stopped: liveness failed.")
        return

    face_result = verify_faces(str(id_path), str(selfie_path))
    _print_step("Face Match", face_result)
    if not face_result.get("verified"):
        print("\nPipeline stopped: face match failed.")
        return

    ocr_result = extract_text(str(id_path))
    _print_step("OCR", ocr_result)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
