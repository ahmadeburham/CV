from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .template_loader import load_templates


MIN_INLIER_RATIO = 0.08
MIN_MATCH_COUNT = 12
ASPECT_RATIO_TOLERANCE = 0.35


def _aspect_ratio(image: np.ndarray) -> float:
    h, w = image.shape[:2]
    return float(w) / float(h) if h else 0.0


def _match_template_orb(upload_img: np.ndarray, template_img: np.ndarray) -> Dict:
    import cv2

    gray_upload = cv2.cvtColor(upload_img, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(gray_upload, None)
    kp2, des2 = orb.detectAndCompute(gray_template, None)

    if des1 is None or des2 is None or not kp1 or not kp2:
        return {
            "score": 0.0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "aspect_ratio_similarity": 0.0,
            "message": "Insufficient keypoints for ORB matching.",
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return {
            "score": 0.0,
            "good_matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "aspect_ratio_similarity": 0.0,
            "message": "Not enough good matches for homography.",
        }

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

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
        }

    path = Path(image_path)
    if not path.exists():
        return {
            "valid": False,
            "template_match_score": 0.0,
            "matched_template": None,
            "message": f"ID image not found: {path}",
        }

    upload_img = cv2.imread(str(path))
    if upload_img is None:
        return {
            "valid": False,
            "template_match_score": 0.0,
            "matched_template": None,
            "message": f"Unreadable ID image: {path}",
        }

    templates = load_templates(template_dir)
    if not templates:
        return {
            "valid": False,
            "template_match_score": 0.0,
            "matched_template": None,
            "message": f"No valid templates found in '{template_dir}'.",
        }

    best_template: Optional[str] = None
    best_result: Dict = {"score": 0.0, "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0, "aspect_ratio_similarity": 0.0}

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
            "good_matches": best_result.get("good_matches", 0),
            "inliers": best_result.get("inliers", 0),
            "inlier_ratio": round(best_result.get("inlier_ratio", 0.0), 4),
            "aspect_ratio_similarity": round(best_result.get("aspect_ratio_similarity", 0.0), 4),
        },
    }
