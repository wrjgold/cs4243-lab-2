"""Stage 1 Localise: board localisation from the reference image and single-frame overlay.

Implement the functions marked ``YOUR CODE HERE`` using your keypoint and
geometry branches. Optional feature families and alternative settings must be
controlled only through ``LocalisationConfig``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .config import LocalisationConfig
from .supplied import to_gray, rect_corners
from .keypoints import detect_corners, describe_keypoints, match_descriptors
from .geometry import ransac_homography, transfer_points, warp_artwork, rectangle_to_rectangle
from . import metrics


def localise_board(reference: np.ndarray, frame: np.ndarray, config: LocalisationConfig
                   ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Find the board in ``frame`` from the rectified ``reference`` image alone.

    Detect and describe keypoints in both images, match with the ratio test,
    estimate the reference-to-frame homography with RANSAC, and project the
    four reference corners (TL, TR, BR, BL) into the frame.

    Returns (H, corners 4 x 2, matches M x 2 (reference index, frame index),
    inlier_mask M, diagnostics). ``diagnostics`` must contain exactly these keys
    (the notebooks and the grading harness read them): ``n_keypoints_ref``,
    ``n_keypoints_frame``, ``n_matches``, ``n_inliers``, ``inlier_ratio``
    (inliers / ratio-test matches), ``median_residual``, ``iterations``,
    ``status``, ``residuals`` (the per-match symmetric transfer errors e_i from
    RANSAC, length M), ``keypoints_ref`` and ``keypoints_frame`` (the arrays the
    match indices refer to). ``median_residual`` is the median over inliers of
    sqrt(e_i / 2), i.e. the RMS one-way transfer error in pixels.
    When localisation fails, ``H`` is None, ``corners`` is NaN and ``status``
    is False.
    """
    # YOUR CODE HERE
    # Convert the reference and frame to grayscale.
    ref_gray, frame_gray = to_gray(reference), to_gray(frame)

    # Detect candidate keypoints in both images.
    ref_points, _ = detect_corners(ref_gray, config.detector)
    frame_points, _ = detect_corners(frame_gray, config.detector)
    # Describe keypoints and retain only points with valid descriptors.
    ref_desc, ref_keep = describe_keypoints(ref_gray, ref_points, config.descriptor)
    frame_desc, frame_keep = describe_keypoints(frame_gray, frame_points, config.descriptor)
    ref_points, frame_points = ref_points[ref_keep], frame_points[frame_keep]

    # Match reference descriptors to frame descriptors using the ratio test.
    matches, _, _ = match_descriptors(ref_desc, frame_desc, config.matcher)

    # Estimate the reference-to-frame homography with RANSAC.
    H, inliers, residuals, iterations, status = ransac_homography(
        ref_points[matches[:, 0]], frame_points[matches[:, 1]], config.ransac)
    # Project the four reference corners when estimation succeeds.
    corners = np.full((4, 2), np.nan)
    if status:
        corners = transfer_points(H, rect_corners(reference.shape[1], reference.shape[0]))
        status = bool(np.isfinite(corners).all())
    # Return no homography and NaN corners when localisation fails.
    if not status:
        H, corners = None, np.full((4, 2), np.nan)

    # Expose counts and residuals for evaluation and visualisation.
    count = int(inliers.sum())
    diagnostics = {
        "n_keypoints_ref": len(ref_points), "n_keypoints_frame": len(frame_points),
        "n_matches": len(matches), "n_inliers": count,
        "inlier_ratio": count / len(matches) if len(matches) else 0.0,
        "median_residual": float(np.median(np.sqrt(residuals[inliers] / 2))) if count else float("inf"),
        "iterations": iterations, "status": status, "residuals": residuals,
        "keypoints_ref": ref_points, "keypoints_frame": frame_points,
    }
    return H, corners, matches, inliers, diagnostics


def artwork_to_frame(artwork: np.ndarray, H: np.ndarray, config: LocalisationConfig) -> np.ndarray:
    """Compose the artwork rectangle with the board-to-frame homography.

    The artwork is placed by mapping its own pixel rectangle onto the
    reference board rectangle (``config.warp.reference_shape``; the artwork's
    own shape when unset) and then applying ``H``. Document in your report how
    the two transformations compose.
    """
    # YOUR CODE HERE
    # Read the artwork and reference sizes, using artwork size as the fallback.
    ah, aw = artwork.shape[:2]
    rh, rw = config.warp.reference_shape if config.warp.reference_shape is not None else (ah, aw)
    # Scale artwork coordinates to the reference, then apply the board homography.
    return H @ rectangle_to_rectangle((aw, ah), (rw, rh))


def overlay_frame(frame: np.ndarray, artwork: np.ndarray, H: np.ndarray,
                  config: LocalisationConfig) -> np.ndarray:
    """Single overlaid frame without occlusion handling.

    Warp the artwork with the composed homography (``artwork_to_frame``) and
    paint it where the coverage mask is set. Returns an RGB float32 frame.
    """
    # YOUR CODE HERE
    # Compose the artwork-to-frame map and warp the artwork.
    warped, coverage = warp_artwork(artwork, artwork_to_frame(artwork, H, config), frame.shape)
    # Copy the frame and replace only pixels covered by the warped artwork.
    result = np.array(frame, dtype=np.float32, copy=True)
    result[coverage] = warped[coverage]
    return result


def evaluate_audit_frames(reference: np.ndarray, frames: Dict[int, np.ndarray],
                          gt_corners: Dict[int, np.ndarray], config: LocalisationConfig) -> Dict:
    """Run ``localise_board`` on annotated frames and summarise with ``metrics``.

    ``frames`` and ``gt_corners`` map frame index -> image / 4 x 2 corners.
    Returns a dict with ``per_frame`` (index -> dict of error, n_matches,
    n_inliers, inlier_ratio, median_residual, status, corners) and ``summary``
    from ``metrics.summarise_errors`` using ``config.success_threshold``.
    """
    # YOUR CODE HERE
    # Evaluate annotated frames that have an image, retaining failed estimates.
    per_frame = {}
    for index in sorted(gt_corners.keys() & frames.keys()):
        # Localise the board and measure its corner error against the annotations.
        _, corners, _, _, diag = localise_board(reference, frames[index], config)

        # Save the frame's error, feature statistics, status, and predicted corners.
        per_frame[index] = {
            "error": metrics.corner_error(corners, gt_corners[index]),
            "n_matches": diag["n_matches"], "n_inliers": diag["n_inliers"],
            "inlier_ratio": diag["inlier_ratio"], "median_residual": diag["median_residual"],
            "status": diag["status"], "corners": corners,
        }

    # Use the supplied metric definitions for the summary.
    summary = metrics.summarise_errors((r["error"] for r in per_frame.values()), config.success_threshold)
    return {"per_frame": per_frame, "summary": summary}


__all__ = ["localise_board", "artwork_to_frame", "overlay_frame", "evaluate_audit_frames"]
