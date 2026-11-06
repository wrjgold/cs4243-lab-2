# CS4243 Lab 2 report
### Wang Ruijie (A0272335H), Sean Tham (A0282350L)

This compact report records how every `YOUR CODE HERE` section is exercised. Elaborate on your implementation and the results you observed; add figures, plots and grounded reasoning that show your understanding. Every reported number must name its clip (common / viewpoint / scale / blur / exit / combined / your clip), frame subset (validation / test / audit) and method variant, and must come from `overlaylab.metrics` or `overlaylab.evaluate`. Thresholds are frozen on the common clip's validation frames; each Stage's **stress table** (`python -m overlaylab.report`, `outputs/stress_tables.md`) reports the common clip (validation and test) and the Stage's two stress clips with that frozen configuration. Paste the generated table into the Stage's section unchanged and explain the row the Stage's question points at; the other rows need no interpretation.

## Stage 1 — Localise
### Code Explanation

#### `keypoints.py`

- `harris_response`:
  - Computes horizontal and vertical gradients using the supplied Sobel operator, then applies Gaussian smoothing to their products to construct the second-moment matrix at each pixel.
  - Calculates the Harris response as `det(M) - kappa * trace(M)^2`, producing a float32 response map with the same height and width as the grayscale image.

- `detect_corners`:
  - Removes non-positive Harris responses, responses below the configured fraction of the maximum, and candidates within the configured image border.
  - In window NMS mode, retains responses equal to the maximum in their square neighbourhood, including ties. In radius NMS mode, greedily accepts candidates in descending response order and suppresses candidates within a circular radius.
  - Returns at most `max_corners` keypoints as `(x, y)` coordinates, ordered by decreasing response, together with their response values.

- `describe_keypoints`:
  - Samples gradients on a centred 16 x 16 grid at the configured patch scale. A gradient-magnitude-weighted orientation histogram determines the dominant orientation; optional Gaussian weighting gives central samples more influence.
  - Rotates the sampling grid to the dominant orientation and expresses gradient directions relative to it. The samples are accumulated into spatial-cell histograms, with each sample's magnitude distributed between neighbouring orientation bins. The standard 4 x 4 cells and 8 bins produce a 128-dimensional descriptor.
  - Normalises each descriptor to unit Euclidean norm, clamps large entries, and normalises it again. Keypoints with out-of-bounds sampling windows or zero gradient energy are discarded. Returns the descriptors and indices of the retained input keypoints.

- `match_descriptors`:
  - Computes pairwise Euclidean distances using descriptor norms and matrix multiplication, then finds the nearest and second-nearest neighbours in the second descriptor set.
  - Accepts a match when the nearest-to-second-nearest distance ratio is below the configured threshold. Ambiguous matches with a zero second-nearest distance are rejected; the optional mutual check also requires the reverse nearest-neighbour match to agree.
  - Returns accepted descriptor-index pairs, nearest-neighbour distances, and distance ratios. Returns empty arrays when there are insufficient descriptors for the ratio test.

#### `geometry.py`

- `bilinear_sample`:
  - Clamps floating-point sample coordinates to the image boundaries and identifies the four neighbouring pixels.
  - Interpolates horizontally and then vertically using the fractional coordinate offsets. Supports grayscale and colour images, with integer coordinates reproducing the corresponding pixel values.

- `normalise_points`:
  - Translates the input points so that their centroid is at the origin, then scales them so that their mean distance from the origin is `sqrt(2)` when the spread is nonzero.
  - Returns the similarity transformation and normalised points. This places coordinates on a comparable numerical scale before constructing the DLT system.

- `dlt_homography`:
  - Normalises source and destination points and constructs the `2N x 9` DLT system, with two equations per correspondence. Uses SVD to obtain the homography from the final right singular vector.
  - Denormalises the estimate using `H = inverse(T_dst) @ H_normalised @ T_src` and scales it so that `H[2, 2] = 1` when possible; otherwise, it uses the matrix norm.
  - Raises an error for fewer than four correspondences or incompatible point-array shapes. Returns a failure status for non-finite inputs, a DLT system with rank below eight, or a non-finite or singular homography. Also returns the DLT singular values for conditioning analysis.

- `transfer_points`:
  - Converts `(x, y)` points to homogeneous coordinates, applies the homography, and divides by the third coordinate to recover image coordinates.
  - Returns an `N x 2` array, leaving projections with non-finite homogeneous coordinates or a zero denominator as `NaN`. Pixel centres use integer coordinates, with `x` indexing columns and `y` indexing rows.

- `symmetric_transfer_error`:
  - Projects source points forward with `H` and destination points backward with `inverse(H)`.
  - Returns the sum of the squared forward and backward distances for each correspondence, in squared pixel units. Singular homographies and invalid projections receive infinite error.

- `ransac_homography`:
  - Uses the configured random generator to sample four correspondences without replacement. Rejects samples containing three nearly collinear points in either image, then estimates a candidate homography using normalised DLT.
  - Classifies inliers using the symmetric transfer error threshold. Selects the model with the most inliers, breaking ties using the smaller total inlier error, and reduces the iteration budget using the best inlier fraction and configured confidence.
  - Refits the homography on the best consensus set and recomputes residuals and inliers. Returns the homography, inlier mask, residuals, iteration count, and status; returns failure if the required inlier count is not met or the refit is invalid.

- `warp_artwork`:
  - Creates a grid of destination pixel centres and applies the inverse artwork-to-frame homography to find their source coordinates.
  - Marks pixels whose source coordinates fall inside the artwork, allowing a small numerical tolerance at the boundaries, and samples those pixels using `bilinear_sample`.
  - Returns the warped float32 image and Boolean coverage mask. Pixels outside coverage remain zero in the warped image.

#### `localisation.py`

- `localise_board`:
  - Converts the reference and frame to grayscale, detects and describes their keypoints, and retains only keypoints with valid descriptors. Matches the descriptors and estimates the reference-to-frame homography using RANSAC.
  - Projects the reference corners in top-left, top-right, bottom-right, bottom-left order. Returns the homography, projected corners, matches, inlier mask, and diagnostics containing feature and match counts, inlier ratio, residuals, iterations, status, and the retained keypoints.
  - Reports the median inlier residual as `median(sqrt(symmetric_error / 2))`. If estimation fails or the projected corners are invalid, returns `H = None`, `NaN` corners, and a false status.

- `artwork_to_frame`:
  - Uses the supplied rectangle-scaling helper to map the artwork's pixel rectangle onto the configured reference rectangle. If the reference shape is unset, uses the artwork's dimensions.
  - Composes this scaling with the reference-to-frame homography as `H_artwork_to_frame = H @ S_artwork_to_reference`. The scaling maps corner pixel centres using dimensions minus one, avoiding a half-pixel offset.

- `overlay_frame`:
  - Obtains the composed artwork-to-frame homography and calls `warp_artwork` to generate the warped artwork and coverage mask.
  - Copies the original frame as float32 and replaces pixels inside coverage with the warped artwork. This Stage 1 overlay does not apply an occlusion mask.

- `evaluate_audit_frames`:
  - Runs `localise_board` on each frame present in both the image and ground-truth dictionaries. Uses the supplied metrics to measure mean corner distance normalised by the longer ground-truth board diagonal.
  - Records per-frame errors, match and inlier statistics, residuals, status, and predicted corners, retaining failed localisations as infinite errors.
  - Returns the per-frame records and a summary containing the median and 90th-percentile finite error, success rate at the configured threshold, total count, and failed count. Failures remain in the success-rate denominator.

### Stress table (Stage 1)
### Question 1 — Matching
### Question 2 — Geometry

## Stage 2 — Track
### Code Explanation

#### `temporal.py`

- `lk_flow`:
  - Builds a square window around each input point and samples the previous frame and its supplied Sobel gradients at the window coordinates. The mean gradient products form a normalised structure tensor; its two eigenvalues measure whether motion is observable in both image directions.
  - Iteratively samples the next frame at the displaced window, solves the two-dimensional Lucas–Kanade update, and stops when the update falls below the configured convergence limit or the iteration limit is reached. It returns a displacement and a mean absolute brightness residual for each point.
  - Marks points invalid when their window leaves either image, the smaller eigenvalue is below the configured minimum, the update does not converge, or the final residual exceeds its limit. The assessed implementation uses the required single-level flow; a pyramid is optional.

- `track_board`:
  - In the independent variant, calls `localise_board` on every frame using the reference image. Each frame receives its own homography, projected corners, status, and monitor statistics.
  - In the flow-assisted variant, localises the first frame and propagates its inlier reference-to-frame correspondences with `lk_flow`. It drops invalid tracks, fits a new homography with RANSAC, and retains the new inlier correspondences for the next frame. The optional corners mode instead propagates four projected corners and requires compatible minimum-point and inlier settings.
  - Re-localises when too few valid points remain, at the configured interval, after a failed homography estimate, or following a monitor warning. Each localiser call is recorded in `reinit_frames`; flow vectors, eigenvalues, residuals, and validity flags are exposed on frames tracked by flow. Frames with no usable model have `lost` status and NaN corners.
  - Builds per-frame statistics with the supplied helper. For flow frames, the inlier ratio is the RANSAC inlier count divided by the number of valid propagated points, and the median residual is the inlier median of the one-way RMS transfer error in pixels.

- `failure_monitor`:
  - Assigns `HIDE` with trigger `no_model` when tracking has no usable model. Otherwise, too few inliers or a low inlier ratio requests `REINIT`, escalating to `HIDE` after the configured number of consecutive weak frames.
  - If inlier checks pass, examines median residual, projected-corner displacement, mean flow, and brightness change in that order. The first finite statistic above its configured maximum triggers `REINIT`; otherwise the frame is `OK`. NaN statistics do not trigger a warning, and an `OK` frame resets the consecutive weak-frame count.

### Stress table (Stage 2)
### Question 3 — Tracking

## Stage 3 — Composite
### Stress table (Stage 3)
### Question 4 — Occlusion

## Stage 4 — Your clip
### Capture and audit
### Overlay Scorecard
### Generative comparison
### Question 5 — Reflection

## Evidence table
| Student file | Function | Report section | Notebook figure / table / video |
|---|---|---|---|
| keypoints.py | harris_response, detect_corners, describe_keypoints, match_descriptors | | |
| geometry.py | bilinear_sample, normalise_points, dlt_homography, symmetric_transfer_error, ransac_homography, transfer_points, warp_artwork | | |
| localisation.py | localise_board, artwork_to_frame, overlay_frame, evaluate_audit_frames | | |
| temporal.py | lk_flow, track_board, failure_monitor | | |
| compositing.py | hysteresis, compensated_difference, occlusion_mask, select_difference_thresholds, composite | | |

## AI-use disclosure table
| Tool / model | Affected artefacts (file, cell, figure, section) | Markers | What the tool contributed | How it was verified |
|---|---|---|---|---|
| | | [AI-CODE] [AI-DESIGN] [AI-VISION] [AI-GENERATION] [AI-LANGUAGE] [HUMAN-DATA] [HUMAN-CHECK] | | |
