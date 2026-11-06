"""Stage 1 Localise, geometry branch.

Implement the functions marked ``YOUR CODE HERE``. Allowed: numpy (including
``numpy.linalg.svd`` and ``numpy.linalg.inv``) and your Lab 1
``bilinear_sample`` (pasted into this file). ``cv2.findHomography`` and ``cv2.warpPerspective``
may appear only in the labelled sanity-check and comparison cells of Notebook 1.

Conventions: points are N x 2 arrays of (x, y) with x indexing columns; a
homography H maps source homogeneous coordinates to destination ones,
x' ~ H x, and is scaled so that H[2, 2] = 1 whenever that entry is not close
to zero. Pixel centres sit at integer coordinates.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import RansacConfig


def bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample a 2-D or H x W x C image at floating-point image coordinates (reuse Lab 1).

    Same signature as Lab 1 (``y`` first, then ``x``); extend it so that a
    colour image returns an array with a trailing channel axis. Coordinates
    are clamped to the valid closed interval, and an exact integer coordinate
    reproduces the corresponding pixel. ``y`` and ``x`` may be arrays of any
    matching shape so that a whole grid can be sampled at once.
    """
    # YOUR CODE HERE
    # Clamp sample coordinates to the image boundaries.
    y = np.clip(np.asarray(y, dtype=float), 0, image.shape[0] - 1)
    x = np.clip(np.asarray(x, dtype=float), 0, image.shape[1] - 1)
    # Find the four neighbouring pixels for each sample.
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, image.shape[0] - 1)
    x1 = np.minimum(x0 + 1, image.shape[1] - 1)
    # Compute interpolation weights and extend them across colour channels.
    dy, dx = y - y0, x - x0
    if image.ndim == 3:
        dy, dx = dy[..., None], dx[..., None]

    # Blend horizontally, then vertically.
    top = (1 - dx) * image[y0, x0] + dx * image[y0, x1]
    bottom = (1 - dx) * image[y1, x0] + dx * image[y1, x1]
    return (1 - dy) * top + dy * bottom


def rectangle_to_rectangle(src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> np.ndarray:
    """Supplied helper: affine scaling that maps a (w, h) pixel rectangle onto another.

    Corner (w-1, h-1) of the source maps to corner (w'-1, h'-1) of the
    destination (pixel-centre convention). Used to compose the artwork
    rectangle with the board-to-frame homography.
    """
    sw, sh = src_size
    dw, dh = dst_size
    return np.array([[(dw - 1) / max(sw - 1, 1), 0, 0],
                     [0, (dh - 1) / max(sh - 1, 1), 0],
                     [0, 0, 1.0]], dtype=np.float64)


def normalise_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation.

    Returns (T, normalised) where T is the 3 x 3 similarity that translates the
    points to zero centroid and scales them so that the mean distance from the
    origin is sqrt(2), and ``normalised`` are the transformed N x 2 points.
    """
    # YOUR CODE HERE
    # Find the centroid and mean distance from it.
    pts = np.asarray(pts, dtype=np.float64)
    centre = pts.mean(axis=0)
    distance = np.linalg.norm(pts - centre, axis=1).mean()
    # Choose the scale for mean distance sqrt(2), avoiding division by zero.
    scale = np.sqrt(2) / distance if distance > np.finfo(float).eps else 1.0

    # Return the translation-and-scale matrix and the transformed points.
    T = np.array([[scale, 0, -scale * centre[0]],
                  [0, scale, -scale * centre[1]], [0, 0, 1]])
    return T, (pts - centre) * scale


def dlt_homography(src: np.ndarray, dst: np.ndarray
                   ) -> Tuple[Optional[np.ndarray], np.ndarray, bool]:
    """Normalised direct linear transform.

    Builds the 2n x 9 system A h = 0 from normalised correspondences, solves
    for h as the right singular vector with the smallest singular value,
    denormalises H = T'^{-1} H_hat T, and scales so that H[2, 2] = 1 when that
    entry is not close to zero.

    Raises ``ValueError`` for fewer than four correspondences. Returns
    (H, singular_values, status); when A is rank deficient (e.g. collinear
    source points) ``status`` is False and ``H`` is None rather than a
    numerically meaningless matrix. ``singular_values`` are those of A, so
    that conditioning can be inspected.
    """
    # YOUR CODE HERE
    # Check that there are at least four finite, matching point pairs.
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("Correspondences must have matching N x 2 shapes")
    if len(src) < 4:
        raise ValueError("At least four correspondences are required")
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        return None, np.empty(0), False

    # Normalise source and destination coordinates before constructing DLT.
    T_src, a = normalise_points(src)
    T_dst, b = normalise_points(dst)

    # Build two DLT equations per correspondence in a 2N x 9 matrix.
    x, y = a.T
    u, v = b.T
    zeros, ones = np.zeros(len(src)), np.ones(len(src))
    A = np.empty((2 * len(src), 9))
    A[0::2] = np.column_stack((-x, -y, -ones, zeros, zeros, zeros, u*x, u*y, u))
    A[1::2] = np.column_stack((zeros, zeros, zeros, -x, -y, -ones, v*x, v*y, v))

    # Four points need the full right-singular basis to expose the null vector.
    _, singular_values, Vt = np.linalg.svd(A, full_matrices=len(src) == 4)
    # Reject systems with too few independent equations for a unique homography.
    tolerance = singular_values[0] * max(A.shape) * np.finfo(float).eps
    if np.count_nonzero(singular_values > tolerance) < 8:
        return None, singular_values, False

    # Undo the coordinate normalisation and fix the matrix scale.
    H = np.linalg.inv(T_dst) @ Vt[-1].reshape(3, 3) @ T_src
    H /= H[2, 2] if abs(H[2, 2]) > 1e-12 else np.linalg.norm(H)
    # Return failure if the resulting homography is non-finite or singular.
    valid = bool(np.isfinite(H).all() and np.linalg.matrix_rank(H) == 3)
    return (H if valid else None), singular_values, valid


def transfer_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to N x 2 points and dehomogenise: pi(H x)."""
    # YOUR CODE HERE
    # Append homogeneous coordinates and apply the homography.
    pts = np.asarray(pts, dtype=float)
    projected = np.column_stack((pts, np.ones(len(pts)))) @ np.asarray(H).T
    # Divide valid projections by their third coordinate; leave invalid ones as NaN.
    result = np.full((len(pts), 2), np.nan)
    valid = np.isfinite(projected).all(axis=1) & (projected[:, 2] != 0)
    result[valid] = projected[valid, :2] / projected[valid, 2:3]
    return result


def symmetric_transfer_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-correspondence e_i = |x' - pi(H x)|^2 + |x - pi(H^-1 x')|^2."""
    # YOUR CODE HERE
    # Measure squared transfer error in both directions.
    try:
        forward = transfer_points(H, src) - dst
        backward = transfer_points(np.linalg.inv(H), dst) - src
    except np.linalg.LinAlgError:
        # A singular homography cannot transfer points backwards.
        return np.full(len(src), np.inf)

    # Sum squared coordinate errors and treat invalid projections as infinite error.
    error = np.sum(forward**2 + backward**2, axis=1)
    return np.where(np.isfinite(error), error, np.inf)


def _three_collinear(p: np.ndarray, tol: float = 1e-6) -> bool:
    """Supplied helper: True if any three of the four points are (nearly) collinear."""
    from itertools import combinations
    scale = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 1.0)
    for i, j, k in combinations(range(len(p)), 3):
        a, b, c = p[i], p[j], p[k]
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area2 < tol * scale * scale:
            return True
    return False


def ransac_homography(src: np.ndarray, dst: np.ndarray, config: RansacConfig
                      ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, int, bool]:
    """Robust homography with adaptive RANSAC and a final refit on all inliers.

    Each iteration draws four correspondences with ``config.rng()``, rejects
    the sample if any three source points are collinear, fits
    ``dlt_homography``, and scores every correspondence by the symmetric
    transfer error; correspondences under ``config.threshold`` are inliers.
    Keep the model with the most inliers (ties: smaller sum of inlier
    residuals); update N = log(1-p) / log(1-(1-eps)^4) from the current inlier
    fraction, capped by ``config.max_iterations``; refit on all inliers.

    Returns (H, inlier_mask, residuals, iterations_used, status) where status
    is False when no model reached ``config.min_inliers`` (H may then be None).
    """
    # YOUR CODE HERE
    # Initialise the best consensus, iteration budget, and seeded random generator.
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    n = len(src)
    best_mask = np.zeros(n, dtype=bool)
    best_residuals = np.full(n, np.inf)
    best_count, best_score = 0, np.inf
    iterations, limit = 0, config.max_iterations
    rng = config.rng()
    # Stop early when there are too few points to sample a homography.
    if n < 4:
        return None, best_mask, best_residuals, iterations, False

    # Draw minimal samples and keep the strongest consensus.
    while iterations < limit:
        iterations += 1
        sample = rng.choice(n, 4, replace=False)

        # Reject collinear samples and fit a candidate homography.
        if _three_collinear(src[sample]) or _three_collinear(dst[sample]):
            continue
        H, _, valid = dlt_homography(src[sample], dst[sample])
        if not valid:
            continue
        # Score every correspondence and select inliers below the threshold.
        residuals = symmetric_transfer_error(H, src, dst)
        mask = residuals < config.threshold
        count, score = int(mask.sum()), float(residuals[mask].sum())
        # Keep more inliers, breaking ties with the smaller total inlier error.
        if count > best_count or (count == best_count and score < best_score):
            best_mask, best_residuals = mask, residuals
            best_count, best_score = count, score

            # Reduce the iteration budget as the estimated inlier fraction improves.
            all_inlier_probability = (count / n)**4
            if all_inlier_probability == 1:
                limit = iterations
            elif all_inlier_probability > 0:
                needed = np.ceil(np.log1p(-config.confidence) /
                                 np.log1p(-all_inlier_probability))
                limit = min(limit, max(1, int(needed)))

    # Reject the run if no consensus reaches the required inlier count.
    if best_count < max(4, config.min_inliers):
        return None, best_mask, best_residuals, iterations, False

    # Refit using the best consensus and recompute its inliers.
    H, _, valid = dlt_homography(src[best_mask], dst[best_mask])
    if not valid:
        return None, best_mask, best_residuals, iterations, False
    # Check the refitted model and return its final residuals, inliers, and status.
    residuals = symmetric_transfer_error(H, src, dst)
    mask = residuals < config.threshold
    valid = bool(mask.sum() >= max(4, config.min_inliers))
    return (H if valid else None), mask, residuals, iterations, valid


def warp_artwork(artwork: np.ndarray, H: np.ndarray, frame_shape: Tuple[int, ...]
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Destination-to-source warp of ``artwork`` into a frame of shape ``frame_shape``.

    ``H`` maps artwork pixel coordinates to frame pixel coordinates. For every
    frame pixel inside the projected artwork quadrilateral, map it back through
    H^{-1} and sample the artwork bilinearly. Returns (warped H x W x C float32,
    coverage H x W bool); pixels that map outside the artwork receive no
    coverage and are zero in ``warped``.
    """
    # YOUR CODE HERE
    # Map destination pixel centres back into the artwork.
    height, width = frame_shape[:2]
    y, x = np.indices((height, width))
    source = transfer_points(np.linalg.inv(H), np.column_stack((x.ravel(), y.ravel())))
    sx, sy = source[:, 0].reshape(height, width), source[:, 1].reshape(height, width)
    # Mark pixels whose inverse projections fall inside the artwork boundaries.
    ah, aw = artwork.shape[:2]
    tolerance = 1e-9
    coverage = (np.isfinite(sx) & np.isfinite(sy) &
                (sx >= -tolerance) & (sx <= aw - 1 + tolerance) &
                (sy >= -tolerance) & (sy <= ah - 1 + tolerance))

    # Sample covered pixels only; leave the background at zero.
    warped = np.zeros((height, width) + artwork.shape[2:], dtype=np.float32)
    warped[coverage] = bilinear_sample(artwork, sy[coverage], sx[coverage])
    return warped, coverage


__all__ = ["bilinear_sample", "rectangle_to_rectangle", "normalise_points", "dlt_homography", "transfer_points",
           "symmetric_transfer_error", "ransac_homography", "warp_artwork"]
