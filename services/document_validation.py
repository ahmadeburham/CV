from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .template_loader import load_templates


MIN_INLIER_RATIO = 0.08
MIN_MATCH_COUNT = 10
ASPECT_RATIO_TOLERANCE = 0.35


def _aspect_ratio(image: np.ndarray) -> float:
    h, w = image.shape[:2]
    return float(w) / float(h) if h else 0.0


def _prepare_for_features(image: np.ndarray) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _resize_to_common_scale(img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    target_max_side = 1400

    def resize(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        max_side = max(h, w)
        if max_side <= target_max_side:
            return img
        scale = target_max_side / float(max_side)
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return resize(img_a), resize(img_b)


def _match_template_orb(upload_img: np.ndarray, template_img: np.ndarray) -> Dict:
    import cv2

    upload_img, template_img = _resize_to_common_scale(upload_img, template_img)

    proc_upload = _prepare_for_features(upload_img)
    proc_template = _prepare_for_features(template_img)

    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=5)
    kp_input, des_input = orb.detectAndCompute(proc_upload, None)
    kp_template, des_template = orb.detectAndCompute(proc_template, None)

    keypoints_input = len(kp_input) if kp_input else 0
    keypoints_template = len(kp_template) if kp_template else 0

    if des_input is None or des_template is None or keypoints_input == 0 or keypoints_template == 0:
        return {
            "score": 0.0,
            "keypoints_template": keypoints_template,
            "keypoints_input": keypoints_input,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "aspect_ratio_similarity": 0.0,
            "message": "Insufficient keypoints for ORB matching.",
        }

    # KNN + ratio test (primary).
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(des_input, des_template, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    # Fallback when ratio test is too strict for low-texture IDs.
    if len(good_matches) < 4:
        cross_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        direct_matches = cross_matcher.match(des_input, des_template)
        direct_matches = sorted(direct_matches, key=lambda x: x.distance)
        good_matches = direct_matches[:60]

    if len(good_matches) < 4:
        upload_ratio = _aspect_ratio(upload_img)
        template_ratio = _aspect_ratio(template_img)
        ratio_delta = abs(upload_ratio - template_ratio)
        aspect_ratio_similarity = max(0.0, 1.0 - min(1.0, ratio_delta / max(template_ratio, 1e-6)))
        return {
            "score": 0.0,
            "keypoints_template": keypoints_template,
            "keypoints_input": keypoints_input,
            "good_matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "aspect_ratio_similarity": float(aspect_ratio_similarity),
            "message": "Not enough good matches for homography.",
        }

    src_pts = np.float32([kp_input[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_template[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(mask.ravel().sum()) if mask is not None else 0
    inlier_ratio = inliers / max(len(good_matches), 1)

    upload_ratio = _aspect_ratio(upload_img)
    template_ratio = _aspect_ratio(template_img)
    ratio_delta = abs(upload_ratio - template_ratio)
    aspect_ratio_similarity = max(0.0, 1.0 - min(1.0, ratio_delta / max(template_ratio, 1e-6)))

    normalized_matches = min(1.0, len(good_matches) / 120.0)
    score = (0.45 * normalized_matches) + (0.45 * inlier_ratio) + (0.10 * aspect_ratio_similarity)

    return {
        "score": float(score),
        "keypoints_template": keypoints_template,
        "keypoints_input": keypoints_input,
        "good_matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
        "aspect_ratio_similarity": float(aspect_ratio_similarity),
        "message": "Template matched with ORB + homography.",
    }


def _is_valid_match(match_result: Dict) -> bool:
    return (
        match_result.get("good_matches", 0) >= MIN_MATCH_COUNT
        and match_result.get("inlier_ratio", 0.0) >= MIN_INLIER_RATIO
        and match_result.get("aspect_ratio_similarity", 0.0) >= (1 - ASPECT_RATIO_TOLERANCE)
    )


def validate_document(image_path: str, template_dir: str = "templates") -> Dict:
    try:
        import cv2
    except Exception as exc:
        return {
            "valid": False,
            "template_match_score": 0.0,
            "matched_template": None,
            "message": f"OpenCV unavailable: {exc}",
            "details": {
                "keypoints_template": 0,
                "keypoints_input": 0,
                "good_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "aspect_ratio_similarity": 0.0,
            },
        }

    path = Path(image_path)
    if not path.exists():
        return {
            "valid": False,
            "template_match_score": 0.0,
            "matched_template": None,
            "message": f"ID image not found: {path}",
            "details": {
                "keypoints_template": 0,
                "keypoints_input": 0,
                "good_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "aspect_ratio_similarity": 0.0,
            },
        }

    upload_img = cv2.imread(str(path))
    if upload_img is None or upload_img.size == 0:
        return {
            "valid": False,
            "template_match_score": 0.0,
            "matched_template": None,
            "message": f"Unreadable ID image: {path}",
            "details": {
                "keypoints_template": 0,
                "keypoints_input": 0,
                "good_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "aspect_ratio_similarity": 0.0,
            },
        }

    templates = load_templates(template_dir)
    if not templates:
        return {
            "valid": False,
            "template_match_score": 0.0,
            "matched_template": None,
            "message": f"No valid templates found in '{template_dir}'.",
            "details": {
                "keypoints_template": 0,
                "keypoints_input": 0,
                "good_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "aspect_ratio_similarity": 0.0,
            },
        }

    best_template: Optional[str] = None
    best_result: Dict = {
        "score": 0.0,
        "keypoints_template": 0,
        "keypoints_input": 0,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "aspect_ratio_similarity": 0.0,
        "message": "No match computed.",
    }

    for template in templates:
        result = _match_template_orb(upload_img, template["image"])
        if result["score"] > best_result["score"]:
            best_result = result
            best_template = template["name"]

    valid = _is_valid_match(best_result)

    return {
        "valid": valid,
        "template_match_score": round(best_result["score"], 4),
        "matched_template": best_template,
        "message": "Document structure matches a known ID template." if valid else "Document failed template structure validation.",
        "details": {
            "keypoints_template": best_result.get("keypoints_template", 0),
            "keypoints_input": best_result.get("keypoints_input", 0),
            "good_matches": best_result.get("good_matches", 0),
            "inliers": best_result.get("inliers", 0),
            "inlier_ratio": round(best_result.get("inlier_ratio", 0.0), 4),
            "aspect_ratio_similarity": round(best_result.get("aspect_ratio_similarity", 0.0), 4),
        },
    }
