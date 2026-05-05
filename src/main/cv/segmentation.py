from __future__ import annotations

from collections import deque

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def threshold_plate_regions(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def region_growing(
    image: np.ndarray,
    seed: tuple[int, int],
    intensity_threshold: int = 25,
) -> np.ndarray:
    gray = to_gray(image)
    h, w = gray.shape
    seed_x, seed_y = seed
    seed_x = max(0, min(w - 1, seed_x))
    seed_y = max(0, min(h - 1, seed_y))

    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
    
    cv2.floodFill(
        gray, 
        mask, 
        (seed_x, seed_y), 
        newVal=255, 
        loDiff=intensity_threshold, 
        upDiff=intensity_threshold, 
        flags=flags
    )
    
    return mask[1:-1, 1:-1]


def edge_contour_candidates(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = to_gray(image)
    edges = cv2.Canny(gray, 80, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 500:
            continue

        aspect = w / max(h, 1)
        if 1.8 <= aspect <= 7.5:
            candidates.append((x, y, x + w, y + h))

    return candidates
