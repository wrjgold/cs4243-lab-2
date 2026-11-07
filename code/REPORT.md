# CS4243 Lab 2 report
### Wang Ruijie (A0272335H), Sean Tham (A0282350L)

The pipeline uses feature matching and planar geometry to localise a board for perspective-aware advertisement placement. Parameters are selected on the common clip's validation frames and fixed for evaluation on its test frames and the supplied stress clips. Quantitative results use `overlaylab.metrics` and `overlaylab.evaluate`, with the clip, frame subset, processing resolution and method identified for each experiment.

## Stage 1 — Localise
### Code Explanation

#### `keypoints.py`

- `harris_response`:
  - Computes horizontal and vertical gradients using the supplied Sobel operator, then applies Gaussian smoothing to their products to construct the second-moment matrix at each pixel.
  - Calculates the Harris response as `det(M) - kappa * trace(M)^2`, producing a float32 response map with the same height and width as the grayscale image.

- `detect_corners`:
  - Removes non-positive Harris responses and responses below the configured fraction of the maximum.
  - In window NMS mode, retains responses equal to the maximum in their square neighbourhood, including ties. In radius NMS mode, greedily accepts candidates in descending response order and suppresses candidates within a circular radius.
  - After NMS, removes candidates within the configured image border. Border candidates participate in suppression before being discarded.
  - Applies the `max_corners` limit before border removal, without replenishing discarded candidates. Returns the surviving keypoints as `(x, y)` coordinates, ordered by decreasing response, together with their response values. The limit is an upper bound on the number of usable descriptors.

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

### Validation choices and coordinate conventions

Stage 1 parameters are selected exclusively on common-clip validation data. Notebook 1 §3 selects NMS mode, `sigma_w` and `kappa` on validation frames 0 and 32 at **640 px**, using maximum repeatability within 3 px and board-corner count to break exact ties. The selected **radius NMS, `sigma_w=2.0`, `kappa=0.06`** gives repeatability **0.881** with **345** board corners in frame 0. These numerical settings are retained in §3.1, which selects NMS radius, corner count and matching settings on all **13 validation frames at 1280 × 720**, with RANSAC seed **0**. Configurations are ranked by descending success rate, then ascending number of unavailable models, finite p90 error and finite median error. Test and stress predictions do not influence selection.

| Common validation, 1280 px; independent localisation | Success at 2% | No model | Finite median error | Finite p90 error |
|---|---:|---:|---:|---:|
| baseline: 600 corners, r=5 | 10/13 | 3 | 0.379% | 0.668% |
| features: 800 corners, r=5 | 11/13 | 2 | 0.392% | 0.680% |
| features: 1000 corners, r=5 | 12/13 | 1 | 0.364% | 0.621% |
| matches: ratio=0.85, mutual=False | 13/13 | 0 | 0.486% | 0.739% |
| matches: ratio=0.85, mutual=True | 13/13 | 0 | 0.440% | 0.718% |
| matches: ratio=0.9, mutual=False | 13/13 | 0 | 0.521% | 0.655% |
| matches: ratio=0.9, mutual=True | 13/13 | 0 | 0.312% | 0.731% |

The baseline returns no model on validation frames 80, 240, 448. Across these frames, its accepted-match counts are 11/20/10 and candidate-inlier counts are 9/9/5, below the required 12 inliers. The nine detector trials combine radii 3/5/7 with corner limits 600/800/1000 to vary feature density and spacing. Eight matcher trials combine ratios 0.75/0.80/0.85/0.90 with both mutual policies, using the highest-ranked detector configuration. Four interaction checks combine the two best detector configurations with the two best matching policies. This staged search limits computation while checking detector–matcher interactions; it does not exhaustively evaluate the joint search space. The selected configuration achieves 13/13 validation successes, with finite median error 0.521% and p90 0.655%. The complete 21-trial settings, per-frame results and summaries are recorded in Notebook 1.

The selected configuration is: Harris `sigma_w=2.0`, `kappa=0.06`, response fraction `0.005`, **radius NMS with radius 5**, **1000 corners**, border 12; descriptor patch scale 1, 36 dominant-orientation bins, 4 × 4 × 8 histograms, Gaussian weighting and clamp 0.2; **ratio 0.90, mutual=False**. RANSAC uses a **9 px²** symmetric-transfer-error threshold, confidence 0.995, cap 2000, minimum 12 inliers and seed 0. Sigma, kappa, NMS mode, descriptor and RANSAC settings are held fixed throughout the §3.1 search.

On validation frames 80/240/448, the selected configuration gives 79/96/85 matches, 29/27/18 inliers and corner errors of 0.658%/0.638%/1.068%. Consensus ratios are 0.367/0.281/0.212. Correspondence support and geometric placement are separate measures; successful localisation alone does not establish that a downstream failure monitor permits rendering. The selected settings are stored in `outputs/stage1_settings.json` and `outputs/thresholds.json` and held fixed for subsequent evaluation.

With ratio 0.90, both mutual policies achieve 13/13 validation successes. The selected mutual=False configuration has p90 0.655% versus 0.731% for the opposite policy; p90 precedes median in the ranking.

Success is defined as **mean corner error ≤ 2% of the longer GT board diagonal**. This fixed tolerance corresponds to approximately **8.42–11.29 px mean corner displacement** across the native validation boards, above the approximately one-pixel annotation noise described in Notebook 1. Finite median and p90 errors exclude unavailable models and must be interpreted alongside coverage. The selection criterion prioritises coverage over finite-estimate accuracy. Selecting sigma and kappa on a two-frame repeatability proxy does not guarantee optimal localisation across all frames; the small validation set limits conclusions about generalisation.

Coordinates are `(x, y)`: **x indexes columns and y indexes rows**. The pixel at row `r`, column `c` is centred at `(c, r)`. Pixel centres are integers; image corner centres range from `(0, 0)` to `(w−1, h−1)`, without a half-pixel offset. Artwork placement composes `H_artwork→frame = H_reference→frame @ S_artwork→reference`, with diagonal scale factors `(W_ref−1)/(W_art−1)` and `(H_ref−1)/(H_art−1)`. The inverse of this composed map supplies source coordinates for bilinear sampling.

![Synthetic checkerboard: known homography and integer pixel-centre convention.](student_files/stage1/warp_convention.png)

The synthetic checkerboard coverage spans x=60–300 and y=40–150, the target quadrilateral's bounds. Notebook 1 §8 compares the bilinear sampler with OpenCV inside coverage. Both use the same pixel-centre convention; interpolation differences occur at projective edges.

Utilities used are the supplied `sobel_gradients`, `gaussian_smooth`, `to_gray`, `rect_corners`, image/video loaders and plotting helpers; `rectangle_to_rectangle` supplies the pixel-centre rectangle scaling. Linear algebra uses NumPy SVD/inversion. Synthetic/GT transforms in the notebooks score correspondences only. The labelled SIFT and whole-frame-warp cells are comparisons, not assessed pipeline calls.

### Stress table (Stage 1)

| row | frames | median err | p90 err | success | inliers (median) | inlier ratio | affine median err | affine success |
|---|---|---|---|---|---|---|---|---|
| common · validation | 13 | 0.52% | 0.65% | 100% | 136 | 0.637 | 0.92% | 92% |
| common · test | 13 | 0.40% | 0.49% | 92% | 151 | 0.654 | 0.84% | 85% |
| viewpoint (coke_perspective) | 32 | 0.44% | 1.02% | 66% | 41 | 0.312 | 3.30% | 19% |
| scale (coke_scale) | 32 | 0.38% | 0.57% | 28% | 5 | 0.079 | 0.67% | 28% |

`localise_board` on every annotated frame with the frozen configuration; *affine* = 6-parameter least-squares fit to the same RANSAC inliers.

The table is generated by `overlaylab.report` using the frozen thresholds at full resolution and saved in `outputs/stress_tables.md`. It measures independent `localise_board` predictions. Both success columns include all eligible annotated frames. Median/p90 errors describe finite estimates only. The following counts distinguish an unavailable model from an estimate outside the tolerance:

| Frozen independent localisation, 1280 px | Eligible frames | No model | Total failures at 2% |
|---|---:|---:|---:|
| common · validation | 13 | 0 | 0 |
| common · test | 13 | 1 | 1 |
| viewpoint (coke_perspective) | 32 | 11 | 11 |
| scale (coke_scale) | 32 | 23 | 23 |

All 21 finite viewpoint estimates satisfy the 2% tolerance. The 11 unavailable models account for all viewpoint localisation failures.

![Common validation and test, native resolution, frozen independent localisation: inlier counts and ratios.](student_files/stage1/native_inliers.png)

Dashed curves show the ratio of RANSAC inliers to accepted ratio-test matches. A high ratio alone does not establish correct board placement; corner error against annotations remains the placement measure.

#### Held-out estimator comparison and worst cases

On the 13 common **test** frames at 1280 px, RANSAC has finite median error 0.399%, p90 0.485%, and success 12/13. DLT on all of the **same ratio-test matches** has median 101.745%, p90 336.153%, and success 0/13. Outliers receive full influence in the all-match algebraic fit; the comparison changes only the estimator. This DLT baseline is the normalised homogeneous least-squares fit, not a nonlinear reprojection-error optimiser.

![Common test, 1280 px, same correspondences: RANSAC against all-match DLT.](student_files/stage1/native_ransac_vs_all.png)

![The two worst native common-test frames: matches, all finite residuals and GT/predicted quadrilaterals.](student_files/stage1/native_worst_cases.png)

- Frame **48**: error unavailable (no model); fails 2%; 9/61 inliers, ratio 0.148, median RMS one-way residual 0.698 px, 2000 RANSAC iterations; model status False.
- Frame **96**: error 0.550%; passes 2%; 223/299 inliers, ratio 0.746, median RMS one-way residual 0.799 px, 28 RANSAC iterations; model status True.

**Common-test failures at 2%: 1/13. Of the two worst cases, 1/2 pass.** The two cases have the largest corner errors among the held-out common-test frames under the frozen configuration. Residual histograms use `log10(1 + e)` to show the full range, with `e` the symmetric transfer error in px² and the frozen 9 px² cutoff marked. The 2% decision is a separate, board-normalised corner measure.

Frame 48 is visibly blurred and has only 9 candidate inliers from 61 accepted matches, below the required 12, so no model is returned. Its candidate-consensus residual does not establish a reliable board-wide homography. Frame 96 contains a card occluding the lower part of the board, but 223 inliers remain from 299 accepted matches on the visible texture. Its 0.550% corner error passes the placement tolerance. Small residuals measure agreement within the selected consensus, while annotated board corners test placement across the whole target.

### Question 1 — Matching

Notebook 1 §5 applies frozen settings to the viewpoint and scale clips at **640 px**. Rate means accepted matches divided by the method's own number of reference descriptors. It measures descriptor acceptance rather than geometric precision or native-resolution localisation success. Both methods use ratio **0.90** with one-way matching (`mutual=False`).

![Viewpoint test, 640 px: best and worst reference/frame matching examples, frozen Harris descriptor.](student_files/stage1/matching_coke_perspective.png)

![Scale test, 640 px: best and worst reference/frame matching examples, frozen Harris descriptor.](student_files/stage1/matching_coke_scale.png)

| Test clip, 640 px | Case/frame | Top-edge angle | Apparent scale | Keystone proxy | Harris reference/frame descriptors | Harris matches/rate |
|---|---|---:|---:|---:|---:|---:|
| coke_perspective | best / 448 | -6.3° | 1.311 | 1.066 | 324/847 | 159 / 0.491 |
| coke_perspective | worst / 256 | 24.9° | 0.956 | 1.566 | 324/900 | 34 / 0.105 |
| coke_scale | best / 80 | 1.5° | 1.060 | 1.015 | 324/816 | 205 / 0.633 |
| coke_scale | worst / 272 | -1.6° | 0.500 | 1.024 | 324/789 | 20 / 0.062 |

For the viewpoint pair, Harris matching falls from **159/324 (49.1%) on frame 448** to **34/324 (10.49%) on frame 256**. The keystone proxy increases from 1.066 to 1.566, while top-edge angle changes from -6.3° to 24.9° and apparent scale from 1.311 to 0.956. For the scale pair, Harris matching falls from **205/324 (63.3%) on frame 80** to **20/324 (6.17%) on frame 272**. Apparent scale falls from 1.060 to 0.500, while top-edge angle remains close to horizontal (1.5° versus -1.6°). The observed drops are consistent with scale and viewpoint sensitivity, although detector repeatability and blur also affect matching.

The descriptor rotates its sampling grid and subtracts the dominant orientation, absorbing approximate **in-plane rotation**. It does not adapt its 16 × 16 support to image scale or undo spatially varying perspective deformation. Under scale change, the same physical texture occupies different histogram cells and the fixed patch covers different structures; under viewpoint change, foreshortening, shear and keystone alter both cell occupancy and gradient orientations beyond one global rotation. Detector repeatability, blur and quantised orientation assignment also prevent perfect invariance.

Apparent scale is the square root of projected board area divided by reference area; it includes foreshortening and does not isolate camera distance. Keystone is the maximum opposite-edge length ratio, a perspective-asymmetry proxy rather than a complete projective-distortion measure.

The labelled SIFT comparison evaluates the **same difficult reference/frame pair** selected for each clip:

| Test clip, 640 px | Difficult frame | Harris acceptance rate | SIFT acceptance rate |
|---|---:|---:|---:|
| coke_perspective | 256 | 10.49% | 16.17% |
| coke_scale | 272 | 6.17% | 29.79% |

On scale frame 272, SIFT accepts **29.79%** of its reference descriptors, compared with **6.17%** for Harris. SIFT selects feature scale as well as orientation, allowing its descriptor support to adapt to image size. On viewpoint frame 256, SIFT accepts **16.17%**, compared with Harris **10.49%**; scale selection does not compensate for general perspective deformation. These results compare complete feature pipelines: SIFT changes both detection and description, and each method has its own reference-descriptor denominator. Acceptance rate alone does not establish geometric correctness or isolate descriptor scale normalisation.

The native scale row tests this limitation with a **fixed reference size and fixed settings**: the reference is not resized to match the board's observed size. Only **9/32 scale frames (28%)** localise successfully, with **23 unavailable models**. The finite median of 0.38% describes the recovered frames. Low coverage demonstrates limited transfer across scale under the fixed-support descriptor. No clip-specific parameter adaptation is applied.

![Common validation frame 176, 640 px: ratio-test rejection examples under the frozen descriptor.](student_files/stage1/ratio_rejections.png)

- Incorrect match correctly rejected on common validation frame 176 (640 px): descriptor indices [58, 364], ratio 0.905, GT-scored transfer distance 20.03 px.
- Correct match rejected on common validation frame 176 (640 px): descriptor indices [104, 431], ratio 0.900, GT-scored transfer distance 3.39 px.

These labels use <4 px one-way transfer only to analyse nearest neighbours. The ratio test measures descriptor distinctiveness, not geometry: near-duplicate letters/edges can make the best and second-best descriptors similarly close, including when the nearest neighbour is geometrically correct. Ground truth labels these examples; it never chooses pipeline matches.

### Question 2 — Geometry

An affine map has six parameters and a constant 2 × 2 linear part: it maps a rectangle to a parallelogram and preserves parallel opposite edges. As the camera translates and rotates around the poster, distance varies across the board and opposite edges converge. A homography's eight degrees of freedom and position-dependent homogeneous denominator represent that keystone. The native viewpoint table compares both models on the **same RANSAC inlier correspondences**. Homography median error is **0.44%** versus affine **3.30%**, with success **21/32** versus **6/32**, respectively. The homography models the orbit more accurately when correspondences support a reliable fit, but insufficient correspondence support leaves 11 viewpoint frames without a model.

![Viewpoint test, 1280 px: frozen homography/affine corner errors over frames and against keystone.](student_files/stage1/native_affine_vs_homography.png)

The representative native viewpoint frame **224** has homography error **1.019%** versus affine error **9.521%**, using identical inliers. Unavailable models count as failures for both methods; the scatter contains finite estimates only. Keystone explains a trend, but local feature coverage and fit noise also affect corner error, so the relationship need not be monotonic.

![Viewpoint test, 1280 px: GT/homography/affine quadrilaterals and the student's single-frame artwork overlay.](student_files/stage1/native_viewpoint_overlay.png)

For the **640-pixel viewpoint test demonstration**, frame **0** is warped toward frame **240** by `H_b @ inverse(H_a)`, using estimated reference-to-frame homographies, not ground-truth initialisation.

![Viewpoint test, 640 px: target, whole-frame board-plane warp and absolute difference; frozen estimated homographies.](student_files/stage1/viewpoint_parallax.png)

The board is approximately aligned while scene structures at other depths shift. With camera translation, the board-plane transform is exact only for that plane; off-plane points have different depth-dependent motion, producing parallax. The difference panel is restricted to valid source overlap; gray unobserved borders are not parallax. Within overlap it also contains interpolation, lighting and estimation error, so not every bright pixel is pure parallax.

This does not harm the overlay because `warp_artwork` samples the artwork only inside projected board coverage and `overlay_frame` copies the original frame elsewhere. The demonstration's whole-frame warp is a labelled comparison, not the rendering method. A **curved/folded or otherwise nonplanar board filmed under camera translation** would place target texture at different depths, so one homography would no longer anchor the complete advertisement correctly. Foreground occlusion is a separate issue handled in Stage 3.

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
  - Builds per-frame statistics with the supplied helper. For flow frames, the inlier ratio is the RANSAC inlier count divided by the number of valid propagated points. The median residual is `median(sqrt(symmetric_transfer_error / 2))` over inliers, combining the forward and backward transfer errors in pixels.

- `failure_monitor`:
  - Assigns `HIDE` with trigger `no_model` when tracking has no usable model. Otherwise, too few inliers or a low inlier ratio requests `REINIT`, escalating to `HIDE` after the configured number of consecutive weak frames. Frames with no usable model also count towards this run.
  - If inlier checks pass, resets the consecutive weak-frame count and examines median residual, projected-corner displacement, mean flow, and brightness change in that order. The first finite statistic above its configured maximum triggers `REINIT`; otherwise the frame is `OK`. NaN statistics do not trigger a warning.

### Stress table (Stage 2)
### Question 3 — Tracking

## Stage 3 — Composite
### Code Explanation

#### `compositing.py`

- `hysteresis`:
  - Validates that the strong and weak masks are two-dimensional and have the same shape, and that connectivity is four or eight.
  - Starts a queue at every strong pixel and visits neighbouring weak pixels until no reachable pixels remain. It returns the reached Boolean mask, leaving disconnected weak responses out.

- `compensated_difference`:
  - Within the warped reference's coverage, ranks pixels by their raw mean absolute colour difference and selects the configured best-agreeing fraction for a shared gain-and-offset fit across RGB channels.
  - Computes the mean absolute colour difference between each covered frame pixel and its gain-and-offset-compensated reference value. It returns a zero-valued map outside coverage, the fitted gain and offset, and an identity fit if coverage is empty.

- `occlusion_mask`:
  - Warps the rectified board reference into the frame with the reference-to-frame homography, then obtains the compensated colour difference inside its coverage.
  - Applies the configured strong and weak thresholds with `hysteresis`, opens the candidate mask with a disc-shaped kernel, and retains connected components meeting `min_area`. The returned mask remains inside board coverage; the difference map and each kept component's label, bounds, and area are returned for diagnosis.

- `select_difference_thresholds`:
  - Precomputes the compensated difference for each supplied frame and searches strong and weak thresholds and minimum component areas. It scores each setting by the mean mask IoU against the supplied validation masks and returns the selected configuration and a log of the candidate scores.
  - Treats scores within an absolute tolerance of `1e-12` of the best score as tied, with zero relative tolerance. Selects the strong threshold nearest the midpoint between the smallest and largest tied strong values, so repeated weak/area combinations do not bias that midpoint. Then favours weak and area values nearest the input configuration, with the smaller strong value as a final tie-breaker.

- `composite`:
  - Forms a visible-board mask from artwork coverage minus occlusion. With feathering enabled, a distance transform reduces alpha near visible-region boundaries; with feathering disabled, alpha is exactly one on visible board pixels and zero elsewhere.
  - Optionally fits a shared gain and offset to the visible artwork and frame pixels, then blends the artwork and original frame with the alpha map. It returns an RGB float32 frame and an alpha map bounded by zero and one.

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
| keypoints.py | harris_response, detect_corners, describe_keypoints, match_descriptors | Stage 1 — Code Explanation; Validation choices; Question 1 | Notebook 1 §§3–5 and SIFT cell; ratio_rejections.png; matching_coke_*.png |
| geometry.py | bilinear_sample, normalise_points, dlt_homography, symmetric_transfer_error, ransac_homography, transfer_points, warp_artwork | Stage 1 — Code Explanation; Coordinate conventions; Estimator comparison; Question 2 | Notebook 1 §§6–8 (executed sanity checks); Notebook 2 §§4–6.1; warp_convention.png; native_worst_cases.png; native_ransac_vs_all.png |
| localisation.py | localise_board, artwork_to_frame, overlay_frame, evaluate_audit_frames | Stage 1 — Code Explanation; Validation choices; Stress table; Questions 1–2 | Notebook 2 §§2–6.1; native_inliers.png; native_viewpoint_overlay.png; outputs/stress_tables.md |
| temporal.py | lk_flow, track_board, failure_monitor | | |
| compositing.py | hysteresis, compensated_difference, occlusion_mask, select_difference_thresholds, composite | | |

## AI-use disclosure table
| Tool / model | Affected artefacts (file, cell, figure, section) | Markers | What the tool contributed | How it was verified |
|---|---|---|---|---|
| | | [AI-CODE] [AI-DESIGN] [AI-VISION] [AI-GENERATION] [AI-LANGUAGE] [HUMAN-DATA] [HUMAN-CHECK] | | |
