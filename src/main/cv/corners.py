from __future__ import annotations

import cv2
import numpy as np

from .segmentation import to_gray


def detect_harris_corners(image: np.ndarray, threshold_ratio: float = 0.01) -> np.ndarray:
    """Harris corner detector.

    Returns:
        Array of shape (N, 2) with (x, y) corner coordinates.
    """
    gray = to_gray(image)
    gray = np.float32(gray)

    response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    response = cv2.dilate(response, None)

    threshold = threshold_ratio * response.max()
    ys, xs = np.where(response > threshold)

    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    corners = np.stack([xs, ys], axis=1).astype(np.int32)
    return corners


def detect_hessian_corners(image: np.ndarray, threshold_ratio: float = 0.005) -> np.ndarray:
    """Hessian-based corner / blob detector.

    Uses the determinant of the Hessian matrix (approximated via
    ``cv2.Sobel`` second-order derivatives) to find corners and blob centres
    that are complementary to the Harris response (Harris misses blob centres;
    Hessian captures them).

    Returns:
        Array of shape (N, 2) with (x, y) corner coordinates.
    """
    gray = to_gray(image).astype(np.float32)

    # Second-order partial derivatives
    Ixx = cv2.Sobel(gray, cv2.CV_64F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(gray, cv2.CV_64F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)

    # Det(H) = Ixx * Iyy - Ixy^2
    det_H = Ixx * Iyy - Ixy ** 2

    # Take positive det (corners / blobs) and normalise
    pos_det = np.clip(det_H, 0, None)
    max_val = pos_det.max()
    if max_val == 0:
        return np.empty((0, 2), dtype=np.int32)

    thresh = threshold_ratio * max_val
    ys, xs = np.where(pos_det > thresh)

    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    corners = np.stack([xs, ys], axis=1).astype(np.int32)
    return corners


def detect_corners(
    image: np.ndarray,
    method: str = "both",
    harris_ratio: float = 0.01,
    hessian_ratio: float = 0.005,
) -> np.ndarray:
    """Combined corner detector.

    Args:
        image:         Input image (RGB or grayscale).
        method:        ``"harris"``, ``"hessian"``, or ``"both"`` (union).
        harris_ratio:  Harris threshold as fraction of max response.
        hessian_ratio: Hessian threshold as fraction of max det(H).

    Returns:
        Array of shape (N, 2) with unique (x, y) corner coordinates.
    """
    parts: list[np.ndarray] = []
    if method in ("harris", "both"):
        parts.append(detect_harris_corners(image, threshold_ratio=harris_ratio))
    if method in ("hessian", "both"):
        parts.append(detect_hessian_corners(image, threshold_ratio=hessian_ratio))

    combined = np.concatenate([p for p in parts if p.size > 0], axis=0) if parts else np.empty((0, 2), dtype=np.int32)
    return combined
