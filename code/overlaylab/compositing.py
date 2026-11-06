"""Stage 3 Composite: occlusion handling and compositing.

Implement the functions marked ``YOUR CODE HERE``. Motion-compensated
differencing is the required occlusion signal; ``mode="colour"`` is an
optional ablation. Permitted library calls: ``cv2.morphologyEx``,
``cv2.connectedComponentsWithStats``, ``cv2.distanceTransform`` (feathering),
and your Lab 1 ``hysteresis``.
"""
from __future__ import annotations

from dataclasses import replace
from collections import deque
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .config import OcclusionConfig, CompositeConfig
from .supplied import to_gray, to_float
from .geometry import warp_artwork
from . import metrics


def hysteresis(strong: np.ndarray, weak: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Keep weak pixels reachable from strong pixels (reuse your Lab 1 implementation).

    ``strong`` and ``weak`` are Boolean masks of the same shape (weak includes
    strong). Traverse with the 4- or 8-connected neighbourhood from every
    strong pixel, through chains of weak pixels, and return the Boolean mask
    of reached pixels. Validate shapes and the connectivity value.
    """
    strong, weak = np.asarray(strong, dtype=bool), np.asarray(weak, dtype=bool)
    if strong.shape != weak.shape or strong.ndim != 2:
        raise ValueError("strong and weak must be same-shape 2-D masks")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    reached = strong.copy()
    queue = deque(map(tuple, np.argwhere(strong)))
    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        neighbours += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    height, width = strong.shape
    while queue:
        y, x = queue.popleft()
        for dy, dx in neighbours:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and weak[ny, nx] and not reached[ny, nx]:
                reached[ny, nx] = True
                queue.append((ny, nx))
    return reached


def compensated_difference(frame: np.ndarray, warped_reference: np.ndarray, coverage: np.ndarray,
                           config: OcclusionConfig) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Per-pixel difference after a global gain/offset fit, zero outside ``coverage``.

    Fit I_frame ~ a * I_ref + b on the ``config.gain_quantile`` fraction of
    covered pixels whose raw difference is smallest (so that illumination
    change alone does not become an occluder), then return the mean absolute
    difference over the colour channels between the frame and the compensated
    reference, together with (a, b). Use colour, not grayscale alone: a
    coloured occluder over a coloured board can have almost the same luminance
    (the ball over the blue board differs by only ~0.13 in grayscale but ~0.3
    in mean channel difference), and the default thresholds assume colour.
    """
    frame = to_float(frame)
    warped_reference = to_float(warped_reference)
    coverage = np.asarray(coverage, dtype=bool)
    difference = np.zeros(coverage.shape, dtype=np.float32)
    if not coverage.any():
        return difference, (1.0, 0.0)

    # Select the best-agreeing covered pixels before fitting one gain and offset
    # shared by all colour channels.
    raw = np.mean(np.abs(frame[coverage] - warped_reference[coverage]), axis=-1)
    n_fit = max(1, int(np.ceil(np.clip(config.gain_quantile, 0, 1) * len(raw))))
    selected = np.argpartition(raw, n_fit - 1)[:n_fit]
    x = warped_reference[coverage][selected].astype(np.float64).ravel()
    y = frame[coverage][selected].astype(np.float64).ravel()
    x_mean, y_mean = x.mean(), y.mean()
    variance = np.mean((x - x_mean) ** 2)
    gain = float(np.mean((x - x_mean) * (y - y_mean)) / variance) if variance > 1e-12 else 1.0
    offset = float(y_mean - gain * x_mean)
    difference[coverage] = np.mean(
        np.abs(frame[coverage] - (gain * warped_reference[coverage] + offset)), axis=-1)
    return difference, (gain, offset)


def _mask_from_difference(difference: np.ndarray, coverage: np.ndarray,
                          config: OcclusionConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Apply hysteresis, opening and component filtering to a cached difference map."""
    seeds = (difference >= config.strong) & coverage
    weak = (difference >= config.weak) & coverage
    mask = hysteresis(seeds, weak, config.connectivity)
    if config.open_radius > 0:
        radius = config.open_radius
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        disc = (xx * xx + yy * yy <= radius * radius).astype(np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, disc).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=config.connectivity)
    kept = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= config.min_area]
    selected = np.isin(labels, kept) & coverage if kept else np.zeros_like(coverage)
    rows = np.array([[i, *stats[i].tolist()] for i in kept], dtype=np.int32).reshape(-1, 6)
    return selected, rows


def occlusion_mask(frame: np.ndarray, H: np.ndarray, reference: np.ndarray, config: OcclusionConfig
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Foreground mask inside the board by motion-compensated differencing.

    1. Warp ``reference`` into the frame with ``H`` (``warp_artwork``) and
       compute the ``compensated_difference`` inside the coverage mask.
    2. Threshold at ``config.strong`` and ``config.weak`` and apply
       ``hysteresis`` with ``config.connectivity``.
    3. Open with a disc of radius ``config.open_radius`` (``cv2.morphologyEx``:
       erosion then dilation, which deletes anything thinner than 2r+1 pixels;
       0 disables) and keep connected components with at least
       ``config.min_area`` pixels.
    4. Return (mask bool H x W, difference map float H x W, component stats).

    ``component stats`` is an array with one row per kept component:
    [label, x, y, width, height, area]. With ``config.mode == "colour"`` an
    appearance model may be used instead (optional ablation).
    """
    if config.mode != "difference":
        raise ValueError(f"Unsupported occlusion mode: {config.mode}")
    warped_reference, coverage = warp_artwork(reference, H, frame.shape)
    difference, _ = compensated_difference(frame, warped_reference, coverage, config)
    mask, stats = _mask_from_difference(difference, coverage, config)
    return mask, difference, stats


def select_difference_thresholds(frames: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                 masks: Sequence[np.ndarray], config: OcclusionConfig
                                 ) -> Tuple[OcclusionConfig, List[Dict]]:
    """Choose strong/weak thresholds and the minimum area on validation masks only.

    ``frames`` is a sequence of (frame, H, reference) triples and ``masks`` the
    matching ground-truth occlusion masks. Grid-search the thresholds and the
    minimum area, score each setting by mean ``metrics.mask_iou`` over the
    validation frames, and return (frozen config, search log) where the log
    holds one dict per setting with its parameters and score.

    Several settings usually tie at the best score: the few validation masks
    do not separate the strong threshold. Break ties towards the middle of the
    tied range of ``strong`` values (not the first or the largest grid point),
    so that the frozen value sits inside the plateau rather than on its edge,
    and state the rule in the report.
    """
    if len(frames) != len(masks) or not frames:
        raise ValueError("frames and masks must be equally sized and nonempty")
    prepared = []
    for (frame, H, reference), target in zip(frames, masks):
        warped, coverage = warp_artwork(reference, H, frame.shape)
        difference, _ = compensated_difference(frame, warped, coverage, config)
        target = np.asarray(target, dtype=bool)
        if target.shape != coverage.shape:
            raise ValueError("ground-truth mask shape differs from frame")
        prepared.append((difference, coverage, target))

    strong_values = sorted(set(np.round(np.r_[np.linspace(0.10, 0.45, 8), config.strong], 4)))
    weak_values = sorted(set(np.round(np.r_[np.linspace(0.04, 0.22, 7), config.weak], 4)))
    area_values = sorted({max(1, int(round(config.min_area * factor))) for factor in (0.5, 1, 2, 4)})
    log = []
    for strong in strong_values:
        for weak in weak_values:
            if weak >= strong:
                continue
            for area in area_values:
                candidate = replace(config, strong=float(strong), weak=float(weak), min_area=area)
                scores = []
                for difference, coverage, target in prepared:
                    predicted, _ = _mask_from_difference(difference, coverage, candidate)
                    score = metrics.mask_iou(predicted, target)
                    if np.isfinite(score):
                        scores.append(score)
                log.append({"strong": float(strong), "weak": float(weak), "min_area": area,
                            "mean_iou": float(np.mean(scores)) if scores else 0.0})

    best_score = max(row["mean_iou"] for row in log)
    tied = [row for row in log if np.isclose(row["mean_iou"], best_score, atol=1e-12)]
    middle_strong = float(np.median([row["strong"] for row in tied]))
    best = min(tied, key=lambda row: (abs(row["strong"] - middle_strong),
                                      abs(row["weak"] - config.weak),
                                      abs(row["min_area"] - config.min_area)))
    return replace(config, strong=best["strong"], weak=best["weak"], min_area=best["min_area"]), log


def composite(frame: np.ndarray, warped: np.ndarray, coverage: np.ndarray, occlusion: np.ndarray,
              config: CompositeConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Alpha-composite the warped artwork into the frame.

    alpha = C * (1 - O) * F, where C is the coverage mask, O the occlusion
    mask and F an optional feathering term that softens the quadrilateral and
    occluder boundaries over ``config.feather`` pixels (0 disables).
    I_out = alpha * W + (1 - alpha) * I. Optional brightness matching of the
    artwork to the board is controlled by ``config.match_gain``.
    Returns (composited RGB float32 frame, alpha map float32 in [0, 1]).
    """
    original = to_float(frame)
    artwork = to_float(warped)
    visible = np.asarray(coverage, dtype=bool) & ~np.asarray(occlusion, dtype=bool)
    alpha = visible.astype(np.float32)
    if config.feather > 0 and visible.any():
        distance = cv2.distanceTransform(visible.astype(np.uint8), cv2.DIST_L2, 5)
        alpha *= np.minimum(distance / (config.feather + 1), 1.0)
    if config.match_gain and visible.any():
        source = artwork[visible].astype(np.float64)
        target = original[visible].astype(np.float64)
        x_mean, y_mean = source.mean(), target.mean()
        variance = np.mean((source - x_mean) ** 2)
        gain = np.mean((source - x_mean) * (target - y_mean)) / variance if variance > 1e-12 else 1.0
        artwork = np.clip(gain * artwork + (y_mean - gain * x_mean), 0, 1).astype(np.float32)
    output = alpha[..., None] * artwork + (1 - alpha[..., None]) * original
    return output.astype(np.float32), alpha


__all__ = ["hysteresis", "compensated_difference", "occlusion_mask", "select_difference_thresholds", "composite"]
