"""Stage 1 Localise, keypoint branch.

Implement the four functions marked ``YOUR CODE HERE``. Do not change the
signatures or return types; the tests and the grading harness call them.
Use only numpy and the helpers in ``overlaylab.supplied`` (Sobel gradients,
Gaussian smoothing, bilinear sampling). OpenCV feature functions
(``cv2.cornerHarris``, ``cv2.SIFT`` and friends) are not allowed here; the
SIFT baseline lives only in the labelled ablation cell of Notebook 1.

Conventions: grayscale images are float32 in [0, 1]; ``x`` indexes columns and
``y`` rows; keypoints are N x 2 arrays of (x, y); descriptors are N x 128.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import DetectorConfig, DescriptorConfig, MatchConfig
from .supplied import sobel_gradients, gaussian_smooth
from .geometry import bilinear_sample


def harris_response(gray: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """Harris response R = det(M) - kappa * tr(M)^2 at every pixel.

    M is the Gaussian-weighted (scale ``config.sigma_w``) second-moment matrix
    built from the supplied normalised Sobel gradients. Returns a float32 map
    with the same shape as ``gray``.
    """
    # Compute horizontal and vertical image gradients.
    gx, gy, _, _ = sobel_gradients(gray)

    # Smooth the gradient products to form the second-moment matrix.
    xx = gaussian_smooth(gx * gx, config.sigma_w)
    yy = gaussian_smooth(gy * gy, config.sigma_w)
    xy = gaussian_smooth(gx * gy, config.sigma_w)

    # Compute the Harris score from the determinant and trace of the matrix.
    return (xx * yy - xy * xy - config.kappa * (xx + yy)**2).astype(np.float32)


def detect_corners(gray: np.ndarray, config: DetectorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Harris corners with non-maximum suppression.

    Steps: compute ``harris_response``; discard responses below
    ``config.response_fraction * max(R)``; apply NMS in mode ``config.nms_mode``
    with radius ``config.nms_radius`` ("window": keep a pixel only if it is the
    maximum of its (2r+1)x(2r+1) neighbourhood; "radius": sort survivors by R
    and greedily accept the strongest while discarding weaker candidates within
    distance r); drop corners within ``config.border`` pixels of the image edge.

    Returns (keypoints, responses): an N x 2 float array of (x, y) ordered by
    decreasing response (N <= config.max_corners) and the matching responses.
    """
    # Check that the configured NMS mode is supported.
    if config.nms_mode not in ("window", "radius"):
        raise ValueError("Unknown NMS mode: " + config.nms_mode)

    # Compute Harris scores and discard non-positive or below-threshold scores.
    R = harris_response(gray, config)
    radius = config.nms_radius
    candidates = (R > 0) & (R >= config.response_fraction * R.max())

    # Remove candidates too close to the image border.
    border = config.border
    if border > 0:
        candidates[:border] = candidates[-border:] = False
        candidates[:, :border] = candidates[:, -border:] = False
    # Extract candidate coordinates and sort them from strongest to weakest.
    y, x = np.nonzero(candidates)
    order = np.argsort(-R[y, x], kind="stable")
    # Window mode: compute the maximum score in each square neighbourhood.
    if config.nms_mode == "window":
        padded = np.pad(R, radius, constant_values=-np.inf)
        maximum = np.full_like(R, -np.inf)
        for dy in range(2 * radius + 1):
            for dx in range(2 * radius + 1):
                np.maximum(maximum, padded[dy:dy + R.shape[0], dx:dx + R.shape[1]],
                           out=maximum)

        # Keep local maxima in score order, including ties, up to the corner limit.
        local_maxima = R[y, x] == maximum[y, x]
        order = order[local_maxima[order]][:config.max_corners]

    # Radius mode: accept strong points and block nearby candidates.
    else:
        blocked = np.zeros_like(candidates)
        accepted = []

        # Build the circular neighbourhood used to suppress nearby points.
        offsets = np.arange(-radius, radius + 1)
        oy, ox = np.meshgrid(offsets, offsets, indexing="ij")
        disk = ox**2 + oy**2 <= radius**2
        # Accept unblocked candidates until the maximum corner count is reached.
        for index in order:
            if len(accepted) >= config.max_corners:
                break
            cy, cx = y[index], x[index]
            if blocked[cy, cx]:
                continue
            accepted.append(index)
            # Suppress the circular neighbourhood of each accepted point.
            yy, xx = cy + oy[disk], cx + ox[disk]
            inside = (yy >= 0) & (yy < R.shape[0]) & (xx >= 0) & (xx < R.shape[1])
            blocked[yy[inside], xx[inside]] = True
        order = np.asarray(accepted, dtype=int)
    # Return (x, y) coordinates and their matching Harris scores.
    return np.column_stack((x[order], y[order])).astype(np.float32), R[y[order], x[order]]


def describe_keypoints(gray: np.ndarray, keypoints: np.ndarray,
                       config: DescriptorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """128-D gradient-histogram descriptors with dominant-orientation normalisation.

    For each keypoint take a 16 x 16 grid of samples spaced ``config.patch_scale``
    pixels apart (so the window is 16 * patch_scale pixels wide),
    build a ``config.orientation_bins``-bin orientation histogram weighted by
    gradient magnitude, assign the dominant orientation, rotate the sampling
    grid so that the first histogram bin is aligned to it, and accumulate a
    ``cells x cells`` grid of ``cell_bins``-bin histograms (4 x 4 x 8 = 128).
    Apply Gaussian spatial weighting when ``config.gaussian_weighting`` is set,
    normalise to unit L2 norm, clamp to ``config.clamp``, and renormalise.

    Keypoints whose (rotated) window leaves the image are dropped. Returns
    (descriptors N' x 128, kept_indices N') where kept_indices indexes the input.
    """
    # Compute image gradients for sampling around each keypoint.
    gx, gy, _, _ = sobel_gradients(gray)
    height, width = gray.shape

    # Build a centred 16 x 16 sampling grid at the configured patch scale.
    size = 16
    grid = (np.arange(size) - (size - 1) / 2) * config.patch_scale
    oy, ox = np.meshgrid(grid, grid, indexing="ij")
    # Optionally give samples near the patch centre more weight.
    weights = np.ones((size, size))
    if config.gaussian_weighting:
        weights = np.exp(-(ox**2 + oy**2) / (2 * (size * config.patch_scale / 2)**2))
    descriptors, kept = [], []
    dimension = config.cells**2 * config.cell_bins

    # Check whether every sample in a patch lies inside the image.
    def inside(x, y):
        return x.min() >= 0 and x.max() <= width - 1 and y.min() >= 0 and y.max() <= height - 1

    # Assign each grid sample to one of the spatial histogram cells.
    cell_y, cell_x = np.indices((size, size))
    cell_y = cell_y * config.cells // size
    cell_x = cell_x * config.cells // size
    # Process each keypoint, skipping patches that cross the image boundary.
    for index, (x, y) in enumerate(keypoints):
        if not inside(x + ox, y + oy):
            continue

        # Assign the dominant gradient orientation around the keypoint.
        px = bilinear_sample(gx, y + oy, x + ox)
        py = bilinear_sample(gy, y + oy, x + ox)
        magnitude = np.hypot(px, py) * weights
        angle = np.mod(np.arctan2(py, px), 2 * np.pi)
        bins = np.floor(angle * config.orientation_bins / (2 * np.pi)).astype(int)
        histogram = np.bincount((bins % config.orientation_bins).ravel(),
                                weights=magnitude.ravel(), minlength=config.orientation_bins)
        dominant = (np.argmax(histogram) + 0.5) * 2 * np.pi / config.orientation_bins

        # Rotate the sampling grid and express gradients relative to that orientation.
        c, s = np.cos(dominant), np.sin(dominant)
        rx, ry = x + c * ox - s * oy, y + s * ox + c * oy
        if not inside(rx, ry):
            continue

        # Sample rotated gradients and subtract the dominant orientation.
        px, py = bilinear_sample(gx, ry, rx), bilinear_sample(gy, ry, rx)
        magnitude = np.hypot(px, py) * weights
        angle = np.mod(np.arctan2(py, px) - dominant, 2 * np.pi)

        # Accumulate cell histograms, blending neighbouring orientation bins.
        orientation = angle * config.cell_bins / (2 * np.pi)
        lower = np.floor(orientation).astype(int)
        fraction = orientation - lower
        base = (cell_y * config.cells + cell_x) * config.cell_bins
        descriptor = np.bincount((base + lower % config.cell_bins).ravel(),
                                 weights=(magnitude * (1 - fraction)).ravel(), minlength=dimension)
        descriptor += np.bincount((base + (lower + 1) % config.cell_bins).ravel(),
                                  weights=(magnitude * fraction).ravel(), minlength=dimension)

        # Normalise, clamp strong bins, then normalise again.
        norm = np.linalg.norm(descriptor)
        if norm == 0:
            continue
        descriptor /= norm
        descriptor = np.minimum(descriptor, config.clamp)
        descriptor /= np.linalg.norm(descriptor)
        # Save the descriptor and its original keypoint index.
        descriptors.append(descriptor)
        kept.append(index)

    # Return descriptors and retained indices, including correctly shaped empty arrays.
    return (np.asarray(descriptors, dtype=np.float32).reshape(-1, dimension),
            np.asarray(kept, dtype=int))


def match_descriptors(desc_a: np.ndarray, desc_b: np.ndarray,
                      config: MatchConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour matching with Lowe's ratio test (optionally mutual).

    For every descriptor in A find the nearest and second-nearest descriptor in
    B by Euclidean distance; accept the match when d1 / d2 < ``config.ratio``.
    With ``config.mutual`` the match must also be the nearest neighbour of the B
    descriptor in A. Returns (pairs M x 2 of (index_a, index_b), distances M,
    ratios M).
    """
    # Convert descriptors to floats and require two neighbours for the ratio test.
    a, b = np.asarray(desc_a, dtype=float), np.asarray(desc_b, dtype=float)
    if len(a) == 0 or len(b) < 2:
        return np.empty((0, 2), dtype=int), np.empty(0), np.empty(0)

    # Compute pairwise distances without an N x M x 128 temporary array.
    squared = np.maximum(np.sum(a*a, axis=1)[:, None] + np.sum(b*b, axis=1)[None, :]
                         - 2 * (a @ b.T), 0)
    # Find the nearest two neighbours and convert squared distances to distances.
    neighbours = np.argsort(squared, axis=1, kind="stable")[:, :2]
    rows = np.arange(len(a))
    d1 = np.sqrt(squared[rows, neighbours[:, 0]])
    d2 = np.sqrt(squared[rows, neighbours[:, 1]])

    # Reject ambiguous matches; two identical nearest neighbours are ambiguous too.
    ratios = np.divide(d1, d2, out=np.ones_like(d1), where=d2 > 0)
    accepted = (d2 > 0) & (ratios < config.ratio)
    # Optionally require each match to be the nearest neighbour in both directions.
    if config.mutual:
        accepted &= np.argmin(squared, axis=0)[neighbours[:, 0]] == rows
    # Return accepted index pairs with their distances and ratios.
    pairs = np.column_stack((rows[accepted], neighbours[accepted, 0]))
    return pairs, d1[accepted], ratios[accepted]


__all__ = ["harris_response", "detect_corners", "describe_keypoints", "match_descriptors"]
