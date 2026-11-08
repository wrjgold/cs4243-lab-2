# Stage 4 reproduction for both group members

The group follows the in-class requirement of one personal clip per student.
Both Sean and Ruijie are required experiments. Each has a 15-second source,
manual audit annotations, two frozen classical videos, an original VACE run and
a staff-permitted descriptive-prompt correction. Neither clip is supplementary.

The supplied staff response in `staff_prompt_correction.png` permits a correction,
requires both runs to be logged, and directs reporting of the corrected output.
The original runs already used aligned binary masks and grey-filled edit regions.
The recorded correction changes the prompt only. All numeric settings, seed,
sampled source indices and input assets stay fixed. The report does not claim
that grey filling was added only on the second run.

## Reproduce the submitted analysis

Run from `code/` in the Python 3.11 environment:

```bash
python student_files/stage4_shared/generative_analysis.py
```

Restart Notebook 4 and run it from top to bottom to reproduce both classical
Scorecards and the complete comparison. This uses the preserved raw videos and
the supplied evaluator. It does not run a generative model or rerender the full
classical videos. Settings in `outputs/thresholds.json` remain fixed.

The complete report is in `REPORT.md`. The two-clip evaluator aggregate is
`outputs/evaluation/stage4_genai_reports.json`. Per-clip reports distinguish
`genai_original` from the corrected `genai` result. The original and corrected
invocation rows are retained unchanged in `manifests/genai_log.csv`.

## Data and outputs

- `data/personal/stage4_sean/` and `data/personal/stage4_ruijie/` contain each
  prepared source, reference, ten audit frames and five masks.
- `data/artwork/sean_explore.png` and `ruijie_orbit.png` are the replacements.
- `outputs/videos/` holds both full-length classical variants per person.
- `outputs/genai/<clip_id>/inputs/` contains the actual input cut, artwork and
  frozen-tracker board masks exported in the fixed input bundle.
- `outputs/genai/<clip_id>/invocations/<invocation_id>/output.mp4` preserves each
  raw VACE output byte for byte. Its selected extracts are stored alongside it.
- `outputs/genai/<clip_id>/output.mp4` and its `frame_XXXXXX.png` files expose the
  corrected result for each person.
- `outputs/genai/<clip_id>/classical_<variant>/` holds the identical classical
  comparison indices.
- `run_records/<clip_id>/<invocation_id>/` preserves the exact input-export
  manifest, actual invocation conditions, log row, metadata and preflight check.
- `input_bundles/` retains the fixed source/mask ZIPs used for generation.
- `classical_summary.json` is the custom classical aggregate. It belongs here,
  while supplied evaluator reports remain in `outputs/evaluation/`.
- `source_baseline/` contains the resolution/frame-rate matched source re-encode
  used to contextualise preservation and timing errors. This is not an edit.

The starter-compatible active paths `manifests/genai_conditions.json`,
`outputs/genai/output.mp4`, `outputs/genai/inputs/`, the root selected extracts
and `outputs/evaluation/genai_frames_report.json` point to Sean's corrected run.
Both students' results remain separately named. Videos from different personal
clips are not concatenated and their scores are not averaged together.

## Original import and fixed sampling

The four downloaded ZIPs were imported with:

```bash
python student_files/stage4_shared/generative_analysis.py --import-results \
  ../stage4_sean_results.zip ../stage4_sean_prompt_revision_results.zip \
  ../stage4_ruijie_results.zip ../stage4_ruijie_prompt_revision_results.zip
```

Use actual archive paths when reproducing that import. Import validates source,
settings and raw-output hashes. The archive SHA256 values and invocation roles
are in `run_records/index.json`. Repeating analysis uses the already imported
records and does not require these external ZIPs.

Each input-export `run.json` retains the initial instruction prompt. The
corrected archive's `conditions.json`, invocation metadata and log row record
the descriptive prompt actually used. These are different stages of the
experiment, not inconsistent claims about which prompt reached the model.

Sean's source indices 226, 227, 270, 376 map to generated indices 0, 1, 23, 80.
Ruijie's 210, 211, 255, 360 map to 0, 1, 24, 80. Extraction uses nominal source
timestamps, preserves the original source labels, and uses cubic resizing.
Every method receives the same full audit dictionary. No favourable-frame
search or ground-truth correction enters generation or the pipeline.

## Model reproduction

`colab_vace_run.ipynb` documents the pinned T4 setup and both recorded prompt
modes. `PROMPT_REVISION=False` selects the original instruction and `True` the
staff-permitted scene description. The modes use separate output directories
and result ZIP names. All actual outputs are already complete, so do not run
another generation merely to improve them.

`handoff.py bundle <clip_id>` is the original source/mask exporter. It now saves
input bundles in `student_files/stage4_shared/input_bundles/` and refuses to
replace an existing fixed bundle. The older `handoff.py import-results` command
handles a single unmodified attempt. Use `generative_analysis.py --import-results`
for this completed original-plus-correction experiment.

## Local backups and submission

Raw recordings, optional WEBM previews, annotation previews and superseded
preparation files have been moved under the repository's ignored
`.local_backups/`. They are excluded from the final ZIP. The supplied datasets,
Python environment, caches and editor files also stay out of the submission.
The ZIP retains both required personal clips, both artworks, all four raw model
outputs, their technical records and the evidence used by the report.
