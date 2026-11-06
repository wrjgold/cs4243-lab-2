"""Stage 2 Track: optical flow, temporal variants, and failure monitoring.

Implement the functions marked ``YOUR CODE HERE``. ``cv2.calcOpticalFlowPyrLK``
and ``cv2.goodFeaturesToTrack`` may appear only in the labelled comparison
cell of Notebook 2. Grayscale images are float32 in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import FlowConfig, MonitorConfig, TrackConfig, LocalisationConfig
from .supplied import to_gray, sobel_gradients, gaussian_pyramid, quad_mask, rect_corners
from .geometry import ransac_homography, transfer_points, bilinear_sample
from .localisation import localise_board

OK, REINIT, HIDE = "OK", "REINIT", "HIDE"


def lk_flow(prev_gray: np.ndarray, next_gray: np.ndarray, points: np.ndarray, config: FlowConfig
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Iterative single-level Lucas-Kanade at sparse points.

    For each point use a (2w+1) x (2w+1) window (``config.window``), the
    supplied Sobel gradients of ``prev_gray``, bilinear sampling of the warped
    window in ``next_gray`` at every iteration, and at most
    ``config.iterations`` updates (stop early when the update is below
    ``config.convergence``). The structure tensor is normalised by the number
    of window pixels so that ``config.min_eigenvalue`` is scale-free.

    Returns (displacements N x 2, eigenvalues N x 2 with lambda_1 >= lambda_2,
    residuals N (mean absolute brightness residual), valid N bool). A point is
    invalid when lambda_2 < config.min_eigenvalue, the residual exceeds
    config.max_residual, the update failed to converge, or the point leaves
    the image. Coarse-to-fine estimation over ``config.pyramid_levels`` is an
    optional extension.
    """
    # Prepare the points and initialise all tracks as invalid.
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(points)
    displacement = np.zeros((n, 2), dtype=np.float64)
    eigenvalues = np.zeros((n, 2), dtype=np.float64)
    residuals = np.full(n, np.inf, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    if n == 0:
        return displacement, eigenvalues, residuals, valid

    # Build a square sampling window around each point.
    height, width = prev_gray.shape
    radius = config.window
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    oy, ox = np.meshgrid(offsets, offsets, indexing="ij")
    x = points[:, 0, None, None] + ox
    y = points[:, 1, None, None] + oy
    # Keep only points whose full window lies inside the previous image.
    inside = (np.isfinite(points).all(axis=1) &
              (points[:, 0] - radius >= 0) & (points[:, 0] + radius < width) &
              (points[:, 1] - radius >= 0) & (points[:, 1] + radius < height))
    if not inside.any():
        return displacement, eigenvalues, residuals, valid

    # Sample the previous image and its gradients at each window.
    gx, gy, _, _ = sobel_gradients(prev_gray)
    ix = bilinear_sample(gx, y[inside], x[inside])
    iy = bilinear_sample(gy, y[inside], x[inside])
    template_all = bilinear_sample(prev_gray, y[inside], x[inside])
    # Form the normalised structure tensor and compute its two eigenvalues.
    a = np.mean(ix * ix, axis=(1, 2))
    b = np.mean(ix * iy, axis=(1, 2))
    c = np.mean(iy * iy, axis=(1, 2))
    spread = np.sqrt((a - c) ** 2 + 4 * b * b)
    eigenvalues[inside, 0] = (a + c + spread) / 2
    eigenvalues[inside, 1] = (a + c - spread) / 2
    # Track only windows with enough texture in both image directions.
    strong = eigenvalues[inside, 1] >= config.min_eigenvalue
    indices = np.flatnonzero(inside)[strong]

    ix, iy, template = ix[strong], iy[strong], template_all[strong]
    a, b, c = a[strong], b[strong], c[strong]
    determinant = a * c - b * b
    converged = np.zeros(n, dtype=bool)
    active = np.ones(len(indices), dtype=bool)
    # Refine each displacement until convergence or the iteration limit.
    for _ in range(config.iterations):
        if not active.any():
            break
        local = np.flatnonzero(active)
        ids = indices[local]
        wx = x[ids] + displacement[ids, 0, None, None]
        wy = y[ids] + displacement[ids, 1, None, None]
        # Stop tracking windows that leave the next image.
        within = ((wx >= 0) & (wx < width) & (wy >= 0) & (wy < height)).all(axis=(1, 2))
        active[local[~within]] = False
        local = local[within]
        if len(local) == 0:
            continue
        ids = indices[local]
        # Compare the warped next image with the template and solve the LK update.
        diff = bilinear_sample(next_gray, wy[within], wx[within]) - template[local]
        rhs_x = np.mean(ix[local] * diff, axis=(1, 2))
        rhs_y = np.mean(iy[local] * diff, axis=(1, 2))
        dx = -(c[local] * rhs_x - b[local] * rhs_y) / determinant[local]
        dy = -(-b[local] * rhs_x + a[local] * rhs_y) / determinant[local]
        step = np.column_stack((dx, dy))
        # Discard failed updates and stop refining points with small updates.
        finite = np.isfinite(step).all(axis=1)
        active[local[~finite]] = False
        local = local[finite]
        ids = indices[local]
        displacement[ids] += step[finite]
        done = np.linalg.norm(step[finite], axis=1) < config.convergence
        converged[ids[done]] = True
        active[local[done]] = False

    # Measure brightness error independently of why a point was rejected.
    ids = np.flatnonzero(inside)
    wx = x[ids] + displacement[ids, 0, None, None]
    wy = y[ids] + displacement[ids, 1, None, None]
    within = ((wx >= 0) & (wx < width) & (wy >= 0) & (wy < height)).all(axis=(1, 2))
    good_ids = ids[within]
    if len(good_ids):
        final = bilinear_sample(next_gray, wy[within], wx[within]) - template_all[within]
        residuals[good_ids] = np.mean(np.abs(final), axis=(1, 2))
        # A sampled point is valid only if it converged with a small residual.
        valid[good_ids] = converged[good_ids] & (residuals[good_ids] <= config.max_residual)
    return displacement, eigenvalues, residuals, valid


@dataclass
class TrackInit:
    """What ``track_board`` needs to (re)initialise from the reference image."""
    reference: np.ndarray
    localisation: LocalisationConfig


@dataclass
class TrackResult:
    homographies: List[Optional[np.ndarray]]
    corners: List[np.ndarray]                 # 4 x 2 per frame (NaN when lost)
    status: List[str]                          # "ok" or "lost"
    stats: List[Dict]                          # consumed by failure_monitor
    reinit_frames: List[int] = field(default_factory=list)
    flow: List[Dict] = field(default_factory=list)  # per-frame vectors, eigenvalues, validity


def _frame_stats(prev_corners, corners, gray_prev, gray, inlier_count, inlier_ratio,
                 median_residual, mean_flow, n_valid, status, n_matches=None):
    """Supplied helper: assemble the per-frame statistics dictionary (``n_matches`` on (re)initialised frames)."""
    disp = float(np.mean(np.linalg.norm(corners - prev_corners, axis=1))) if (
        prev_corners is not None and np.all(np.isfinite(corners)) and np.all(np.isfinite(prev_corners))) else float("nan")
    if gray_prev is not None and np.all(np.isfinite(corners)):
        m = quad_mask(gray.shape, corners)
        bright = float(np.mean(np.abs(gray[m] - gray_prev[m]))) if m.any() else float("nan")
    else:
        bright = float("nan")
    return {"inlier_count": int(inlier_count), "inlier_ratio": float(inlier_ratio),
            "median_residual": float(median_residual), "corner_displacement": disp,
            "mean_flow": float(mean_flow), "brightness_change": bright,
            "n_valid_points": int(n_valid), "status": status,
            "n_matches": (int(n_matches) if n_matches is not None else None)}


def track_board(frames: Sequence[np.ndarray], init: TrackInit, config: TrackConfig) -> TrackResult:
    """Track the board through a sequence with one of two variants.

    ``config.variant == "independent"``: run ``localise_board`` on every frame
    from the reference image.

    ``config.variant == "flow_assisted"``: localise frame 0, then propagate the
    inlier correspondences (or the corners, ``config.propagate``) to the next
    frame with ``lk_flow``, drop invalid points, re-estimate the homography
    with RANSAC between the reference points and the propagated points, and
    reinitialise from the reference image when fewer than
    ``config.reinit_min_points`` valid points remain, every
    ``config.reinit_every`` frames (0 disables), or when RANSAC fails.

    Optional corner propagation has four points only: use reinit_min_points <= 4,
    RANSAC min_inliers = 4 and monitor min_inlier_count <= 4 if selecting it.

    Both variants return a ``TrackResult`` with a homography, projected
    corners, a status flag and the statistics the failure monitor consumes
    for every frame (build them with the supplied ``_frame_stats``; use
    ``inlier_ratio`` = inliers / valid propagated points for the flow variant
    and ``median_residual`` = median over inliers of sqrt(e_i / 2) px). The
    flow-assisted variant also stores, for every frame after the first, a dict
    in ``TrackResult.flow[t]`` with keys ``points`` (tracked points N x 2),
    ``vectors`` (displacements N x 2), ``eigenvalues`` (N x 2), ``residuals``
    (N) and ``valid`` (N bool); frames that were (re)initialised store {}.
    """
    # Check that the tracking variant and propagation mode are supported.
    if config.variant not in ("independent", "flow_assisted"):
        raise ValueError(f"Unknown tracking variant: {config.variant}")
    if config.propagate not in ("inliers", "corners"):
        raise ValueError(f"Unknown propagation mode: {config.propagate}")

    # Initialise the results and the reference-to-frame point correspondences.
    result = TrackResult([], [], [], [])
    ref_corners = rect_corners(init.reference.shape[1], init.reference.shape[0])
    ref_points = np.empty((0, 2), dtype=np.float64)
    frame_points = np.empty((0, 2), dtype=np.float64)
    previous_gray = previous_corners = None
    reinit_next = False

    def relocalise(frame, index, gray):
        nonlocal ref_points, frame_points
        # Find the board from the reference and record the localiser call.
        H, corners, matches, inliers, diag = localise_board(init.reference, frame, init.localisation)
        result.reinit_frames.append(index)
        # Start new tracks from the projected corners or matched inliers.
        if H is not None:
            if config.propagate == "corners":
                ref_points, frame_points = ref_corners.copy(), corners.copy()
            else:
                pairs = matches[inliers]
                ref_points = diag["keypoints_ref"][pairs[:, 0]].copy()
                frame_points = diag["keypoints_frame"][pairs[:, 1]].copy()
        else:
            # Clear the tracked points when localisation fails.
            ref_points = np.empty((0, 2), dtype=np.float64)
            frame_points = np.empty((0, 2), dtype=np.float64)
        # Build the status and monitor statistics for the localised frame.
        status = "ok" if H is not None else "lost"
        stats = _frame_stats(previous_corners, corners, previous_gray, gray,
                             diag["n_inliers"], diag["inlier_ratio"],
                             diag["median_residual"], float("nan"),
                             diag["n_inliers"], status, diag["n_matches"])
        return H, corners, status, stats

    for index, frame in enumerate(frames):
        gray = to_gray(frame)
        flow_record = {}
        # Localise independently, at fixed intervals, or when tracking needs recovery.
        needs_reinit = (config.variant == "independent" or index == 0 or
                        previous_gray is None or len(frame_points) < config.reinit_min_points or
                        (config.reinit_every > 0 and index % config.reinit_every == 0) or
                        reinit_next)
        if needs_reinit:
            H, corners, status, stats = relocalise(frame, index, gray)
        else:
            # Propagate the tracked points with LK and save the flow diagnostics.
            vectors, eigenvalues, residuals, valid = lk_flow(previous_gray, gray, frame_points, config.flow)
            tracked = frame_points + vectors
            flow_record = {"points": tracked.copy(), "vectors": vectors, "eigenvalues": eigenvalues,
                           "residuals": residuals, "valid": valid}
            # Drop invalid tracks and re-localise if too few points remain.
            source, destination = ref_points[valid], tracked[valid]
            if len(source) < max(4, config.reinit_min_points):
                H, corners, status, stats = relocalise(frame, index, gray)
                flow_record = {}
            else:
                # Fit a new reference-to-frame homography from the propagated points.
                H, inliers, errors, _, ok = ransac_homography(source, destination, init.localisation.ransac)
                if not ok:
                    H, corners, status, stats = relocalise(frame, index, gray)
                    flow_record = {}
                else:
                    # Project the board corners and reject non-finite estimates.
                    corners = transfer_points(H, ref_corners)
                    if not np.isfinite(corners).all():
                        H, corners, status, stats = relocalise(frame, index, gray)
                        flow_record = {}
                    else:
                        # Keep only RANSAC inliers for the next frame.
                        status = "ok"
                        ref_points, frame_points = source[inliers].copy(), destination[inliers].copy()
                        count = int(inliers.sum())
                        # Record the inlier ratio, transfer residual and mean flow.
                        stats = _frame_stats(previous_corners, corners, previous_gray, gray,
                                             count, count / len(source),
                                             float(np.median(np.sqrt(errors[inliers] / 2))),
                                             float(np.mean(np.linalg.norm(vectors[valid], axis=1))),
                                             len(source), status)

        # Save this frame's estimate and diagnostics.
        result.homographies.append(H)
        result.corners.append(corners)
        result.status.append(status)
        result.stats.append(stats)
        result.flow.append(flow_record)
        # Request localisation on the next frame if the monitor reports trouble.
        recent = result.stats[-max(1, config.monitor.hide_after_consecutive):]
        reinit_next = failure_monitor(recent, config.monitor)[-1][0] != OK
        previous_gray, previous_corners = gray, corners
    return result


def failure_monitor(stats: Sequence[Dict], config: MonitorConfig) -> List[Tuple[str, str]]:
    """Decide OK / REINIT / HIDE for every frame from the per-frame statistics.

    ``stats`` is the list produced by ``track_board``: dictionaries with
    ``inlier_count``, ``inlier_ratio``, ``median_residual``,
    ``corner_displacement``, ``mean_flow``, ``brightness_change`` and
    ``status``. Returns a list of (flag, trigger) where ``trigger`` names the
    statistic (or "none") responsible for the decision. Because a homography
    is defined only up to scale, compare projected-corner displacement and
    reprojection error, never homography matrices.

    Required policy (the tests check it; the thresholds are yours to set):
    1. ``status != "ok"`` -> HIDE with trigger "no_model";
    2. ``inlier_count`` < min_inlier_count or ``inlier_ratio`` < min_inlier_ratio
       -> REINIT with trigger "inlier_count" / "inlier_ratio", escalating to
       HIDE once this has happened on ``hide_after_consecutive`` consecutive
       frames (frames under rule 1 count towards the run);
    3. otherwise the first of ``median_residual``, ``corner_displacement``,
       ``mean_flow``, ``brightness_change`` that exceeds its maximum -> REINIT
       with that statistic as trigger (NaN statistics never fire);
    4. otherwise OK with trigger "none"; an OK frame resets the run.
    """
    # Set up the weak-frame counter and the ordered statistic limits.
    decisions = []
    consecutive = 0
    maxima = (("median_residual", config.max_median_residual),
              ("corner_displacement", config.max_corner_displacement),
              ("mean_flow", config.max_mean_flow),
              ("brightness_change", config.max_brightness_change))
    for frame in stats:
        # Hide immediately when no usable model is available.
        if frame["status"] != "ok":
            consecutive += 1
            decisions.append((HIDE, "no_model"))
            continue
        # Check the inlier count before the inlier ratio.
        trigger = None
        if frame["inlier_count"] < config.min_inlier_count:
            trigger = "inlier_count"
        elif frame["inlier_ratio"] < config.min_inlier_ratio:
            trigger = "inlier_ratio"
        # Repeated weak frames escalate from reinitialisation to hiding.
        if trigger:
            consecutive += 1
            decisions.append((HIDE if consecutive >= config.hide_after_consecutive else REINIT, trigger))
            continue
        # With enough inliers, check the remaining statistics in priority order.
        consecutive = 0
        for name, maximum in maxima:
            value = frame[name]
            if np.isfinite(value) and value > maximum:
                trigger = name
                break
        # Request reinitialisation for a breached limit; otherwise accept the frame.
        decisions.append((REINIT, trigger) if trigger else (OK, "none"))
    return decisions


__all__ = ["OK", "REINIT", "HIDE", "lk_flow", "TrackInit", "TrackResult", "track_board", "failure_monitor"]
