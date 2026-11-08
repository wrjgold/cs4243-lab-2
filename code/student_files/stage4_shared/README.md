# Stage 4: separate Sean and Ruijie runs

Both clips are prepared and their manual audit annotations pass the checker.
Sean's source frames and annotations are unchanged. Ruijie's reference uses the
outer white border of the flat front face, excluding the side panel. Do not
modify either prepared video now that annotation is complete.

## Current files and submission

Four active helper files are needed: this README, `handoff.py`,
`colab_vace_run.ipynb` and `classical_analysis.py`. Keep them while finishing Stage 4. They are not individually
named mandatory deliverables, but they reproduce input export and the model run.
The assignment requests the whole `code/` folder and permits added helper scripts
under `student_files/`.

`preparation_archive.zip` preserves the nine superseded files from
`stage4_sean/` and `stage4_ruijie/`, including conversion scripts, original
preparation hashes and geometry, synthetic setup records and the old setup-only
notebook. Every member was hash-verified before its working copy was removed.
This archive is a local backup and is not needed in the submission. To restore
the old files, extract it into `code/student_files/`.

Keep and submit each chosen clip's prepared `clip.mp4`, `reference.png`, ten
audit frame PNGs and five masks under `data/personal/<clip_id>/`, plus its
replacement PNG under `data/artwork/`. Keep the capture/audit manifests, active
generative conditions and the per-person conditions under `outputs/genai/`.
The completed submission also needs both classical variants and their statistics,
the actual raw generative output and invocation log, evaluator reports,
Scorecards and the completed report/notebooks. Those outputs are not replaced by
these preparation files.

Do not include raw backup recordings/photos under `data/stage4_sean/` or
`data/stage4_ruijie/`, annotation-check previews, caches, `.DS_Store`, the old
setup archive or the supplied data. These backups and previews stay on the laptop.
Git ignore rules do not automatically exclude files from a manually created ZIP.

## What each person receives

Complete notebook in Sean's Drive: https://colab.research.google.com/drive/1vDx_YdlaJGKZDOl4a19jP9lik0kMBIZL

The Drive copy is shared with Wang Ruijie as Editor, as requested.
General access remains Restricted. Ruijie should save his own copy in Drive
before running, so both people keep independent notebook versions. He can also
upload the local `colab_vace_run.ipynb` into his own account.

Send `colab_vace_run.ipynb` from this directory, or its uploaded Colab link,
and that person's input ZIP after the final pipeline results exist. Each person
opens the notebook in their own Google account, saves a copy in Drive and chooses
a free T4 GPU. Their sessions and account limits are separate. No paid plan or
Hugging Face token is required. The first cell uploads one input ZIP. Subsequent
cells prepare the inputs, load VACE, perform one actual generation and download
the raw result and logs. There are no GPU benchmark cells.

The old Sean setup-only notebook is historical. Use the complete notebook in
this directory for both people. It encodes the new explicit replacement prompt,
so old cached wall-poster prompt embeddings must not be reused.

## Before producing input ZIPs

Confirm that the final Stage 1-3 code and `outputs/thresholds.json` have been
integrated into the current branch. A thresholds file existing is not evidence
that the friend has finished Stage 3. Do not tune settings on either personal
clip. Run the required classical variants from `code/`:

```bash
../.venv/bin/python -m overlaylab.run_pipeline --clip stage4_sean --variant independent --artwork sean_explore --thresholds outputs/thresholds.json
../.venv/bin/python -m overlaylab.run_pipeline --clip stage4_sean --variant flow_assisted --artwork sean_explore --thresholds outputs/thresholds.json
../.venv/bin/python -m overlaylab.run_pipeline --clip stage4_ruijie --variant independent --artwork ruijie_orbit --thresholds outputs/thresholds.json
../.venv/bin/python -m overlaylab.run_pipeline --clip stage4_ruijie --variant flow_assisted --artwork ruijie_orbit --thresholds outputs/thresholds.json
```

## Produce the two separate model input bundles

From `code/`:

```bash
../.venv/bin/python student_files/stage4_shared/handoff.py bundle stage4_sean
../.venv/bin/python student_files/stage4_shared/handoff.py bundle stage4_ruijie
```

The helper reads each person's fixed `outputs/genai/<clip_id>/conditions.json`.
It checks annotations and verifies that the pipeline statistics match the
current code, source, reference and frozen settings. It exports the chosen cut
and one binary board-quadrilateral mask per source frame, using the same
construction as Notebook 4. Lost tracking produces an empty mask, without
ground-truth correction. Audit masks and audit corner coordinates are not
included in the upload ZIP.

The input ZIPs are written to:

- `code/outputs/genai/stage4_sean/stage4_sean_inputs.zip`
- `code/outputs/genai/stage4_ruijie/stage4_ruijie_inputs.zip`

Keep the fixed bundles after generation. The script refuses to overwrite an
existing input ZIP. It never runs the classical pipeline or a video model.

## Run Colab independently

1. Open the complete notebook and use File > Save a copy in Drive.
2. Select Runtime > Change runtime type > T4 GPU. Free availability is not guaranteed.
3. Run from the top. Upload only your own input ZIP when prompted.
4. Check the printed clip ID, displayed artwork and greyed source previews.
5. Let the setup and single generation complete. The settings are 832 x 480,
   81 output frames at 16 fps, 20 steps, seed 0, guidance 5 and conditioning scale 1.
   The prompt is `Replace the artwork on the board in this video with the attached artwork.`
6. Download `<clip_id>_results.zip` using the last cell. Save it before disconnecting.

White board-mask regions are resized with nearest-neighbour interpolation and
grey-filled after the video is resized. VACE receives the new artwork as its
reference image, not the original-board reference. The model itself generates
the output pixels. The notebook never pastes artwork into the result.

Keep the first completed output even if it looks poor. On a technical error,
run the final download cell to preserve the log and failure record. Only an
unchanged technical retry is permitted, with both invocations preserved. A new
runtime cannot recover files lost when an old runtime was disconnected.

## Import and evaluate the results on Sean's laptop

From `code/`, use the actual downloaded ZIP paths:

```bash
../.venv/bin/python student_files/stage4_shared/handoff.py import-results "$HOME/Downloads/stage4_sean_results.zip"
../.venv/bin/python student_files/stage4_shared/handoff.py import-results "$HOME/Downloads/stage4_ruijie_results.zip"
```

The helper preserves the raw MP4 unchanged in each person's output folder,
retains per-invocation records and merges actual invocation rows into the
required `manifests/genai_log.csv`. It extracts the selected generative frames
at nominal timestamps, resizes the extracts to source resolution and retains
their original source indices. It also extracts the same indices from both
classical videos when present. It does not search for visually better frames.

Use the existing evaluator in frames mode. For Ruijie:

```bash
../.venv/bin/python -m overlaylab.evaluate --frames outputs/genai/stage4_ruijie/frame_*.png --clip stage4_ruijie --artwork ruijie_orbit --out outputs/evaluation/stage4_ruijie_genai_frames_report.json
../.venv/bin/python -m overlaylab.evaluate --frames outputs/genai/stage4_ruijie/classical_independent/frame_*.png --clip stage4_ruijie --artwork ruijie_orbit --out outputs/evaluation/stage4_ruijie_independent_frames_report.json
../.venv/bin/python -m overlaylab.evaluate --frames outputs/genai/stage4_ruijie/classical_flow_assisted/frame_*.png --clip stage4_ruijie --artwork ruijie_orbit --out outputs/evaluation/stage4_ruijie_flow_assisted_frames_report.json
```

For Sean, replace `stage4_ruijie` with `stage4_sean` and `ruijie_orbit` with
`sean_explore`. The evaluator loads the same complete audit dictionary for
each method's selected frames. Finish each Scorecard and the Notebook 4/report
analysis from the actual results. The assignment's active
`manifests/genai_conditions.json` remains Sean's, while each person's complete
conditions are preserved next to their separate outputs. To use the starter
Notebook 4 for a different active clip, copy that person's conditions into the
active manifest and preserve the other person's archived outputs first.

## Complete the assignment's Notebook 4

The helper does not replace the required notebook analysis. Part I evaluates
both full-length classical videos and provides the Scorecard to complete with
measured values. Part II exports the model inputs, extracts the generated
comparison frames, checks the log and evaluates the same selected frames for
both classical variants and VACE. Finish the comparison table, staff-oracle
discussion and Question 5 reflection from the actual evidence.

The starter notebook reads one active clip and expects the active generative
video at `outputs/genai/output.mp4`. After importing a person's results, copy
that person's conditions and raw video into the starter's active locations.
For Ruijie, from `code/`:

```bash
cp outputs/genai/stage4_ruijie/conditions.json manifests/genai_conditions.json
cp outputs/genai/stage4_ruijie/output.mp4 outputs/genai/output.mp4
```

For Sean, use `stage4_sean` in both commands. Restart the Notebook 4 kernel and
run in order. Do not run model generation again. The starter's input-export
cell can run normally even if the helper already built the fixed input ZIP.
Its frame-extraction cell removes stale active PNGs before extracting the
selected indices. The helper uses the same cubic resizing for its extracts.
The archived raw video and conditions remain intact in each person's folder.
Preserve each person's evaluator reports and notebook evidence before switching
the active clip, since the starter uses shared report filenames.

VACE uses a short cut because 81 output frames at 16 fps span five seconds.
Both classical variants still process the full 15-second source clip. The
comparison frames and temporal pair remain fixed inside that cut. The board
masks come from flow-assisted statistics because Notebook 4 Part II section 1
explicitly uses `<clip_id>_flow_assisted_stats.json`, rather than selecting a
variant after inspecting personal-clip results.

## Verification performed during preparation

Prepared-media metadata and hashes, both annotation sets, notebook syntax and
cell separation were checked. A synthetic local harness exercised ZIP upload,
timestamp sampling, grey filling, a fake model call, invocation logging,
download/import and original-index extraction. It also checked that a completed
output cannot be regenerated accidentally. This is not a real GPU run or an
artwork-quality check. These preparation checks did not run personal generation.

The active handoff and classical-analysis files are versioned so both people can run them from
their own checkout. The preparation archive and original recordings remain
ignored. Actual per-person conditions, required manifests, generation outputs
and invocation logs are retained.

## Completed classical analysis

Notebook 4 now includes a separate `(Our Code)` classical section for both clips.
It uses `classical_analysis.py`, saves the two completed Scorecards and four
evaluator reports, and generates four evidence/timeline figures used in REPORT.md.
It also prepares both classical variants on the fixed selected indices, with
separate frames-mode reports for the later generative comparison.
The starter cells and frozen algorithms remain intact. The supplied placeholder
Scorecard is overwritten by the measured per-clip Scorecards in the added section.
Only the classical notebook cells have been executed at this stage.
