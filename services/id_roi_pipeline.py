from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re

import cv2
import numpy as np

try:
    from paddleocr import PaddleOCR
except Exception:  # noqa: BLE001
    PaddleOCR = None

try:
    import easyocr
except Exception:  # noqa: BLE001
    easyocr = None


TEMPLATE_SIZE = (1280, 853)  # width, height

TEMPLATE_ROIS = {
    "header_noise": [420, 15, 1030, 170],
    "photo": [25, 55, 390, 570],
    "name_line_1": [800, 175, 1225, 255],
    "name_line_2": [730, 250, 1235, 355],
    "birthplace": [670, 375, 1215, 495],
    "address": [690, 470, 1185, 590],
    "birth_date": [55, 595, 470, 775],
    "id_number": [575, 600, 1250, 780],
    "name_combined": [730, 175, 1235, 355],
    "text_block_combined": [670, 175, 1235, 590],
}

VARIABLE_REGIONS_FOR_VALIDATION = [
    "header_noise",
    "photo",
    "name_line_1",
    "name_line_2",
    "birthplace",
    "address",
    "birth_date",
    "id_number",
    "name_combined",
    "text_block_combined",
]

OCR_FIELDS = ["name_line_1", "name_line_2", "birthplace", "address", "birth_date", "id_number"]
NUMERIC_FIELDS = {"birth_date", "id_number"}

_PADDLE_READER = None
_EASY_READER = None


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None or img.size == 0:
        raise ValueError(f"Unable to read image: {path}")
    return img


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def detect_card_quadrilateral(image: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 180)

    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(morphed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    debug_contours = image.copy()
    cv2.drawContours(debug_contours, contours[:20], -1, (0, 255, 255), 2)

    img_area = image.shape[0] * image.shape[1]
    best_quad = None

    for contour in contours[:50]:
        area = cv2.contourArea(contour)
        if area < img_area * 0.10:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            best_quad = approx.reshape(4, 2).astype("float32")
            break

    quad_overlay = image.copy()
    if best_quad is not None:
        cv2.polylines(quad_overlay, [best_quad.astype(int)], True, (0, 255, 0), 3)

    return best_quad, {"contours": debug_contours, "detected_quad": quad_overlay, "edges": morphed}


def crop_and_rectify_card(image: np.ndarray, quad: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    rect = _order_points(quad)
    (tl, tr, br, bl) = rect

    max_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    max_width = max(max_width, 400)
    max_height = max(max_height, 250)

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return cv2.resize(warped, target_size, interpolation=cv2.INTER_CUBIC)


def align_to_template(card_image: np.ndarray, template_image: np.ndarray) -> Tuple[np.ndarray, bool]:
    card_gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(card_gray, None)
    kp2, des2 = orb.detectAndCompute(template_gray, None)

    if des1 is not None and des2 is not None and len(kp1) > 20 and len(kp2) > 20:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=2)

        good = []
        for pair in matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) >= 20:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None and mask is not None and mask.sum() >= 12:
                aligned = cv2.warpPerspective(card_image, H, (template_image.shape[1], template_image.shape[0]))
                return aligned, True

    try:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
        _, warp = cv2.findTransformECC(template_gray, card_gray, warp, cv2.MOTION_AFFINE, criteria)
        aligned = cv2.warpAffine(
            card_image,
            warp,
            (template_image.shape[1], template_image.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        return aligned, True
    except Exception:  # noqa: BLE001
        return cv2.resize(card_image, (template_image.shape[1], template_image.shape[0])), False


def get_template_rois(template_size: Tuple[int, int] = TEMPLATE_SIZE) -> Dict[str, List[int]]:
    src_w, src_h = TEMPLATE_SIZE
    dst_w, dst_h = template_size
    sx = dst_w / src_w
    sy = dst_h / src_h

    scaled = {}
    for name, (x1, y1, x2, y2) in TEMPLATE_ROIS.items():
        scaled[name] = [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]
    return scaled


def get_validation_mask(template_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = template_shape[:2]
    rois = get_template_rois((w, h))

    mask = np.full((h, w), 255, dtype=np.uint8)
    for region in VARIABLE_REGIONS_FOR_VALIDATION:
        x1, y1, x2, y2 = rois[region]
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask


def validate_card_against_template(
    aligned_card: np.ndarray,
    template_image: np.ndarray,
    validation_mask: np.ndarray,
    min_score: float = 0.58,
) -> Dict[str, object]:
    aligned_gray = cv2.cvtColor(aligned_card, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    masked_aligned = cv2.bitwise_and(aligned_gray, aligned_gray, mask=validation_mask)
    masked_template = cv2.bitwise_and(template_gray, template_gray, mask=validation_mask)

    mask_bool = validation_mask > 0
    if not np.any(mask_bool):
        return {
            "template_match": False,
            "template_score": 0.0,
            "method": "ssim_orb_masked",
            "ssim_like": 0.0,
            "orb_score": 0.0,
            "error": "Validation mask removed entire image.",
            "debug_images": {
                "validation_mask": validation_mask,
                "masked_template": masked_template,
                "masked_aligned_card": masked_aligned,
            },
        }

    diff = np.abs(masked_aligned.astype(np.float32) - masked_template.astype(np.float32))
    mean_diff = float(np.mean(diff[mask_bool]))
    ssim_like = max(0.0, 1.0 - (mean_diff / 255.0))

    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(masked_aligned, validation_mask)
    kp2, des2 = orb.detectAndCompute(masked_template, validation_mask)

    orb_score = 0.0
    if des1 is not None and des2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for pair in matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        orb_score = min(1.0, len(good) / 120.0)

    final_score = round(0.55 * ssim_like + 0.45 * orb_score, 4)
    return {
        "template_match": final_score >= min_score,
        "template_score": final_score,
        "method": "ssim_orb_masked",
        "ssim_like": round(ssim_like, 4),
        "orb_score": round(orb_score, 4),
        "debug_images": {
            "validation_mask": validation_mask,
            "masked_template": masked_template,
            "masked_aligned_card": masked_aligned,
        },
    }


def _refine_roi(roi: np.ndarray) -> np.ndarray:
    if roi is None or roi.size == 0:
        return roi
    h, w = roi.shape[:2]
    pad_x = int(w * 0.03)
    pad_y = int(h * 0.05)
    x1 = max(0, pad_x)
    y1 = max(0, pad_y)
    x2 = max(x1 + 1, w - pad_x)
    y2 = max(y1 + 1, h - pad_y)
    return roi[y1:y2, x1:x2]


def crop_rois(aligned_card: np.ndarray, rois: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
    crops = {}
    for name, (x1, y1, x2, y2) in rois.items():
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(aligned_card.shape[1], x2)
        y2 = min(aligned_card.shape[0], y2)
        crop = aligned_card[y1:y2, x1:x2].copy()
        crops[name] = _refine_roi(crop)
    return crops


def preprocess_text_roi(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(up, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(denoised)
    blur = cv2.GaussianBlur(clahe, (3, 3), 0)
    sharp = cv2.addWeighted(clahe, 1.8, blur, -0.8, 0)
    return cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)


def preprocess_numeric_roi(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
    up = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(up, 7, 30, 30)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(denoised)
    return cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _get_paddle_reader() -> Optional[object]:
    global _PADDLE_READER
    if _PADDLE_READER is not None:
        return _PADDLE_READER
    if PaddleOCR is None:
        return None
    try:
        _PADDLE_READER = PaddleOCR(use_angle_cls=True, lang="arabic", show_log=False)
    except Exception:  # noqa: BLE001
        return None
    return _PADDLE_READER


def _get_easy_reader() -> Optional[object]:
    global _EASY_READER
    if _EASY_READER is not None:
        return _EASY_READER
    if easyocr is None:
        return None
    try:
        _EASY_READER = easyocr.Reader(["ar", "en"], gpu=False)
    except Exception:  # noqa: BLE001
        return None
    return _EASY_READER


def _ocr_paddle(image: np.ndarray) -> str:
    reader = _get_paddle_reader()
    if reader is None:
        return ""
    result = reader.ocr(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image, cls=True)
    lines: List[str] = []
    if result and result[0]:
        for box in result[0]:
            if len(box) > 1 and box[1]:
                lines.append(str(box[1][0]).strip())
    return " ".join([x for x in lines if x])


def _ocr_easy(image: np.ndarray, allowlist: Optional[str] = None) -> str:
    reader = _get_easy_reader()
    if reader is None:
        return ""
    out = reader.readtext(image, detail=0, paragraph=False, allowlist=allowlist)
    return " ".join([str(x).strip() for x in out if str(x).strip()])


def _clean_text(value: str) -> str:
    value = value.replace("|", " ").replace("_", " ")
    return re.sub(r"\s+", " ", value).strip()


def _clean_digits(value: str) -> str:
    value = value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    return re.sub(r"[^0-9/-]", "", value)


def run_ocr_on_rois(crops: Dict[str, np.ndarray]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, np.ndarray]]:
    raw_ocr: Dict[str, str] = {}
    cleaned: Dict[str, str] = {}
    preprocessed: Dict[str, np.ndarray] = {}

    for field in OCR_FIELDS:
        roi = crops.get(field)
        if roi is None or roi.size == 0:
            raw_ocr[field] = ""
            cleaned[field] = ""
            continue

        if field in NUMERIC_FIELDS:
            proc = preprocess_numeric_roi(roi)
            text = _ocr_paddle(proc) or _ocr_easy(proc, allowlist="0123456789٠١٢٣٤٥٦٧٨٩/-")
            text_clean = _clean_digits(text)
        else:
            proc = preprocess_text_roi(roi)
            text = _ocr_paddle(proc) or _ocr_easy(proc)
            text_clean = _clean_text(text)

        preprocessed[field] = proc
        raw_ocr[field] = text
        cleaned[field] = text_clean

    return raw_ocr, cleaned, preprocessed


def _build_overlay(image: np.ndarray, rois: Dict[str, List[int]]) -> np.ndarray:
    overlay = image.copy()
    for name, (x1, y1, x2, y2) in rois.items():
        color = (0, 0, 255) if name == "header_noise" else (0, 255, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, name, (x1 + 5, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return overlay


def save_debug_outputs(
    output_dir: Path,
    raw_card_crop: Optional[np.ndarray],
    rectified_card: Optional[np.ndarray],
    aligned_card: Optional[np.ndarray],
    crops: Dict[str, np.ndarray],
    preprocessed: Dict[str, np.ndarray],
    debug_images: Dict[str, np.ndarray],
    rois: Dict[str, List[int]],
) -> None:
    card_dir = output_dir / "card"
    crops_dir = output_dir / "crops"
    pre_dir = output_dir / "preprocess"
    debug_dir = output_dir / "debug"

    for folder in [card_dir, crops_dir, pre_dir, debug_dir, output_dir / "ocr"]:
        folder.mkdir(parents=True, exist_ok=True)

    if raw_card_crop is not None:
        cv2.imwrite(str(card_dir / "raw_card_crop.jpg"), raw_card_crop)
    if rectified_card is not None:
        cv2.imwrite(str(card_dir / "rectified_card.jpg"), rectified_card)
    if aligned_card is not None:
        cv2.imwrite(str(card_dir / "aligned_card.jpg"), aligned_card)
        cv2.imwrite(str(debug_dir / "roi_overlay.jpg"), _build_overlay(aligned_card, rois))

    for name, crop in crops.items():
        if crop is not None and crop.size > 0:
            cv2.imwrite(str(crops_dir / f"{name}.jpg"), crop)

    for name, prep in preprocessed.items():
        if prep is not None and prep.size > 0:
            cv2.imwrite(str(pre_dir / f"{name}_preprocessed.jpg"), prep)

    for name, img in debug_images.items():
        if img is not None and img.size > 0:
            cv2.imwrite(str(debug_dir / f"{name}.jpg"), img)


def save_json_results(output_dir: Path, result: Dict) -> None:
    ocr_dir = output_dir / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    with open(ocr_dir / "results.json", "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)


def _combine_fields(cleaned: Dict[str, str]) -> Dict[str, Optional[str]]:
    full_name = _clean_text(f"{cleaned.get('name_line_1', '')} {cleaned.get('name_line_2', '')}") or None
    id_num = re.sub(r"\D", "", cleaned.get("id_number", ""))
    dob = cleaned.get("birth_date") or None

    if len(id_num) >= 14:
        match = re.search(r"\d{14}", id_num)
        id_num = match.group(0) if match else id_num[:14]
    elif not id_num:
        id_num = None

    return {
        "full_name": full_name,
        "birthplace": cleaned.get("birthplace") or None,
        "address": cleaned.get("address") or None,
        "birth_date": dob,
        "id_number": id_num,
    }


def process_id_image(image_path: str, template_path: str, output_dir: str) -> Dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result: Dict = {
        "input_image": image_path,
        "template_image": template_path,
        "template_size": [TEMPLATE_SIZE[0], TEMPLATE_SIZE[1]],
        "card_detected": False,
        "alignment_success": False,
        "template_validation": {
            "template_match": False,
            "template_score": 0.0,
            "method": "ssim_orb_masked",
        },
        "fields": {
            "full_name": None,
            "birthplace": None,
            "address": None,
            "birth_date": None,
            "id_number": None,
        },
        "raw_ocr": {
            "name_line_1": "",
            "name_line_2": "",
            "birthplace": "",
            "address": "",
            "birth_date": "",
            "id_number": "",
        },
    }

    debug_images: Dict[str, np.ndarray] = {}

    try:
        image = load_image(image_path)
        template = cv2.resize(load_image(template_path), TEMPLATE_SIZE, interpolation=cv2.INTER_CUBIC)
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        save_json_results(out, result)
        return result

    quad, det_debug = detect_card_quadrilateral(image)
    debug_images.update(det_debug)

    if quad is None:
        save_debug_outputs(out, None, None, None, {}, {}, debug_images, get_template_rois())
        result["error"] = "Failed to detect card boundary from raw scene image."
        save_json_results(out, result)
        return result

    result["card_detected"] = True

    raw_card_crop = crop_and_rectify_card(image, quad, TEMPLATE_SIZE)
    rectified_card = raw_card_crop.copy()

    aligned_card, alignment_success = align_to_template(rectified_card, template)
    result["alignment_success"] = alignment_success

    validation_mask = get_validation_mask(template.shape)
    validation = validate_card_against_template(aligned_card, template, validation_mask)
    result["template_validation"] = {
        "template_match": bool(validation.get("template_match", False)),
        "template_score": float(validation.get("template_score", 0.0)),
        "method": str(validation.get("method", "ssim_orb_masked")),
    }

    debug_images.update(validation.get("debug_images", {}))

    rois = get_template_rois((aligned_card.shape[1], aligned_card.shape[0]))
    crops = crop_rois(aligned_card, rois)

    raw_ocr, cleaned, preprocessed = run_ocr_on_rois(crops)
    result["raw_ocr"] = raw_ocr
    result["fields"] = _combine_fields(cleaned)

    if not result["template_validation"]["template_match"]:
        result["error"] = "Template validation failed: card design does not match expected template."

    save_debug_outputs(out, raw_card_crop, rectified_card, aligned_card, crops, preprocessed, debug_images, rois)
    save_json_results(out, result)
    return result
