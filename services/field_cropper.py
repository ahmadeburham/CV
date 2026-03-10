from __future__ import annotations

from typing import Dict

import numpy as np


EGYPTIAN_ID_LAYOUT = {
    "photo": (0.03, 0.17, 0.34, 0.88),
    "birth_date": (0.05, 0.64, 0.42, 0.86),
    "name": (0.48, 0.19, 0.97, 0.45),
    "address": (0.47, 0.44, 0.97, 0.66),
    "id_number": (0.46, 0.66, 0.98, 0.86),
    "full_card_text": (0.02, 0.10, 0.98, 0.95),
}


def _crop_by_ratio(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    h, w = image.shape[:2]
    left = max(0, min(w, int(w * x1)))
    top = max(0, min(h, int(h * y1)))
    right = max(0, min(w, int(w * x2)))
    bottom = max(0, min(h, int(h * y2)))

    if right <= left or bottom <= top:
        return np.zeros((1, 1, 3), dtype=image.dtype)

    return image[top:bottom, left:right].copy()


def crop_egyptian_id_fields(rectified_image: np.ndarray) -> Dict[str, np.ndarray]:
    if rectified_image is None or rectified_image.size == 0:
        return {}

    fields = {}
    for field_name, (x1, y1, x2, y2) in EGYPTIAN_ID_LAYOUT.items():
        fields[field_name] = _crop_by_ratio(rectified_image, x1, y1, x2, y2)

    return fields
