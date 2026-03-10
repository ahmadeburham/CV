from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class DocumentRectificationError(Exception):
    """Raised when document rectification fails."""


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    max_width = max(max_width, 400)
    max_height = max(max_height, 250)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _find_card_quadrilateral(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_area = image.shape[0] * image.shape[1]

    for contour in contours[:30]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.15:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")

    return None


def rectify_document(image: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, str]]:
    """
    Rectifies a document using contour-based card detection and perspective transform.
    Returns (rectified_image, info_dict).
    """
    if image is None or image.size == 0:
        return None, {"message": "Input image is empty or unreadable."}

    try:
        quad = _find_card_quadrilateral(image)
        if quad is None:
            return None, {"message": "Failed to detect card boundary quadrilateral."}

        rectified = _four_point_transform(image, quad)
        if rectified is None or rectified.size == 0:
            return None, {"message": "Perspective transform returned empty output."}

        h, w = rectified.shape[:2]
        if h > w:
            rectified = cv2.rotate(rectified, cv2.ROTATE_90_CLOCKWISE)

        return rectified, {"message": "Card rectification successful."}
    except Exception as exc:  # noqa: BLE001
        return None, {"message": f"Document rectification error: {exc}"}
