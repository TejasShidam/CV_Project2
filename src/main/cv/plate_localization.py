from __future__ import annotations

import cv2
import numpy as np

from .corners import detect_harris_corners
from .segmentation import edge_contour_candidates, region_growing, threshold_plate_regions


def _clip_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(1, min(width, x2))
    y2 = max(1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, width, height)
    return (x1, y1, x2, y2)


def _corner_score(corners: np.ndarray, box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    if corners.size == 0:
        return 0

    inside_x = (corners[:, 0] >= x1) & (corners[:, 0] <= x2)
    inside_y = (corners[:, 1] >= y1) & (corners[:, 1] <= y2)
    return int((inside_x & inside_y).sum())


def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _best_plate_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = image.shape[:2]
    threshold_mask = threshold_plate_regions(image)
    seed = (w // 2, h // 2)
    grown_mask = region_growing(image, seed)

    candidates = edge_contour_candidates(image)

    threshold_box = _mask_to_bbox(threshold_mask)
    if threshold_box is not None:
        candidates.append(threshold_box)

    grown_box = _mask_to_bbox(grown_mask)
    if grown_box is not None:
        candidates.append(grown_box)

    if not candidates:
        return None

    corners = detect_harris_corners(image)

    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for cand in candidates:
        x1, y1, x2, y2 = _clip_bbox(*cand, width=w, height=h)
        area = max(1, (x2 - x1) * (y2 - y1))
        corner_density = _corner_score(corners, (x1, y1, x2, y2)) / area
        aspect = (x2 - x1) / max((y2 - y1), 1)

        aspect_score = 1.0 - min(abs(aspect - 3.5) / 3.5, 1.0)
        score = 0.6 * corner_density + 0.4 * aspect_score
        scored.append((score, (x1, y1, x2, y2)))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _maybe_perspective_correct(crop: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(np.float32)
    width = int(max(rect[1][0], rect[1][1]))
    height = int(min(rect[1][0], rect[1][1]))
    if width <= 10 or height <= 10:
        return None

    dst = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(box, dst)
    corrected = cv2.warpPerspective(crop, matrix, (width, height))
    if corrected.size == 0:
        return None
    return corrected


def localize_plate_region_with_bbox(
    image: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    bbox = _best_plate_bbox(image)
    if bbox is None:
        return image, (0, 0, w, h)

    x1, y1, x2, y2 = _clip_bbox(*bbox, width=w, height=h)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image, (0, 0, w, h)

    corrected = _maybe_perspective_correct(crop)
    if corrected is not None:
        return corrected, (x1, y1, x2, y2)

    return crop, (x1, y1, x2, y2)


def localize_plate_region(image: np.ndarray) -> np.ndarray:
    crop, _ = localize_plate_region_with_bbox(image)
    return crop
