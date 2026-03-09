from pathlib import Path
from typing import List


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_template_paths(directory: Path) -> List[Path]:
    if not directory.exists():
        return []

    # Prefer cleaned templates when available.
    cleaned_dir = directory / "cleaned"
    if cleaned_dir.exists():
        cleaned = [p for p in sorted(cleaned_dir.glob("*")) if p.suffix.lower() in VALID_EXTENSIONS]
        if cleaned:
            return cleaned

    direct = [p for p in sorted(directory.glob("*")) if p.suffix.lower() in VALID_EXTENSIONS]
    if direct:
        return direct

    return [p for p in sorted(directory.rglob("*")) if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]


def load_templates(template_dir: str = "templates") -> List[dict]:
    """Load template images from disk, preferring cleaned structural templates."""
    try:
        import cv2
    except Exception:
        return []

    directory = Path(template_dir)
    template_paths = _collect_template_paths(directory)

    templates = []
    for path in template_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue

        templates.append({"name": path.name, "path": str(path), "image": image})

    return templates
