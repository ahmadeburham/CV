from pathlib import Path
from typing import List


def load_templates(template_dir: str = "templates") -> List[dict]:
    """Load ID template images from disk."""
    try:
        import cv2
    except Exception:
        return []

    directory = Path(template_dir)
    if not directory.exists():
        return []

    templates = []
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        image = cv2.imread(str(path))
        if image is None:
            continue

        templates.append({"name": path.name, "path": str(path), "image": image})

    return templates
