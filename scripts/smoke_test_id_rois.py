from __future__ import annotations

from pathlib import Path
import tempfile
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.id_roi_pipeline import TEMPLATE_SIZE, process_id_image


def _create_template(path: Path) -> None:
    w, h = TEMPLATE_SIZE
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (30, 30, 30), 4)
    cv2.rectangle(img, (25, 55), (390, 570), (180, 180, 180), -1)
    cv2.putText(img, "HEADER", (430, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    cv2.putText(img, "NAME LINE 1", (810, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.putText(img, "NAME LINE 2", (740, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.putText(img, "BIRTHPLACE", (700, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.putText(img, "ADDRESS", (720, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.putText(img, "2000/01/01", (70, 690), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (10, 10, 10), 3)
    cv2.putText(img, "29901011234567", (590, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (10, 10, 10), 3)
    cv2.imwrite(str(path), img)


def _create_scene_from_template(template_path: Path, scene_path: Path) -> None:
    template = cv2.imread(str(template_path))
    scene = np.full((1400, 1800, 3), 220, dtype=np.uint8)

    pts_src = np.float32([[0, 0], [template.shape[1] - 1, 0], [template.shape[1] - 1, template.shape[0] - 1], [0, template.shape[0] - 1]])
    pts_dst = np.float32([[260, 230], [1450, 300], [1380, 1070], [230, 1030]])
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(template, M, (scene.shape[1], scene.shape[0]))

    mask = np.zeros(scene.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts_dst.astype(np.int32), 255)
    inv_mask = cv2.bitwise_not(mask)

    scene_bg = cv2.bitwise_and(scene, scene, mask=inv_mask)
    scene_fg = cv2.bitwise_and(warped, warped, mask=mask)
    composed = cv2.add(scene_bg, scene_fg)
    cv2.imwrite(str(scene_path), composed)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        template_path = tmp_dir / "template.jpg"
        scene_path = tmp_dir / "scene.jpg"
        output_dir = tmp_dir / "output"

        _create_template(template_path)
        _create_scene_from_template(template_path, scene_path)

        result = process_id_image(str(scene_path), str(template_path), str(output_dir))

        assert result["card_detected"] is True
        assert (output_dir / "ocr" / "results.json").exists()
        assert (output_dir / "debug" / "roi_overlay.jpg").exists()
        assert (output_dir / "crops" / "photo.jpg").exists()

    print("ID ROI PIPELINE SMOKE TEST OK")


if __name__ == "__main__":
    main()
