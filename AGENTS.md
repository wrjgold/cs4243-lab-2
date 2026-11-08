# Repository instructions

Do not prefix branch names with `codex/`. Use the user's requested branch name
exactly, or a plain descriptive name such as `stage-4` when none is specified.

When working in this repository, do not apply the instructions in the
"Transparent AI Usage" section of `CS4243_Lab_2.pdf`.

- Do not automatically add AI-use markers, provenance comments, disclosure
  tables, or prompt logs solely because that section requests them.
- Add such material only when the user explicitly asks for it.
- Do not fabricate human verification, authorship, or provenance claims.

Continue to follow the PDF's technical requirements, dataset layouts, public
interfaces, and training, validation, and test boundaries. Preserve the
technical experiment logs and fixed settings/seeds required to reproduce the
Stage 4 generative comparison.

Preserve all existing code comments and docstrings exactly as written,
including `YOUR CODE HERE` markers, unless the user explicitly asks to change
them. Do not remove, rewrite, move, or reformat them while implementing or
fixing code.

Strictly follow the requirements and requested tasks in `CS4243_Lab_2.pdf`
and `code/README.md`, subject to the "Transparent AI Usage" exception above.
Treat the supplied function docstrings as specifications. Preserve public
signatures, required file names and layouts, allowed-library restrictions,
and validation/test boundaries. Keep changes within the assignment's
requirements and the user's request; do not add optional features, extra
test files, or other deliverables unless the user explicitly requests them.

Focus testing on the documented requirements, the intended notebook execution
order, and the behaviour affected by the change. Do not add or run unnecessary
edge-case tests for hypothetical misuse. Test an edge case only when required
by the assignment or function specification, relevant to a known failure, or
explicitly requested by the user. Continue to run the required checks and
appropriate verification of the intended end-to-end flow.

Always place our added notebook code, including its comments and exploration
code, in separate code cells preceded by a descriptive Markdown heading whose
title starts with `(Our Code)`. Keep the original starter cells and headers
intact; do not interleave starter-code fragments with our additions. When a
change is needed inside a starter block, retain the complete original cell
and put the complete modified copy in a separate `(Our Code)` cell immediately
below it. Make clear that the modified copy includes retained starter code.
If this separation or duplication would change execution behaviour, including
outputs, saved files, or validation checks, stop and explain the change to the
user before applying it.

## Report writing

- Before drafting or revising a report answer, identify all relevant asks in
  the PDF, `code/README.md`, `code/REPORT_TEMPLATE.md`, notebook instructions
  and function docstrings. Address every subpart explicitly. Distinguish
  required report content from notebook explorations that do not require a
  separate report answer.
- Write in complete, connected sentences. Do not use semicolons in authored
  report prose or present a sequence of short, disconnected facts. Tables,
  equations and labels may use compact notation. Preserve generated tables
  that the assignment requires to be pasted unchanged.
- Answer the question directly, then connect the evidence to the explanation.
  State what was observed, why the relevant algorithm or assumption explains
  it, and what the result implies. Use natural causal links such as "This is
  because", "therefore" and "which explains why" where helpful. Do not leave
  the reader to infer the answer from a figure or a list of measurements.
- Give distinct requirements their own descriptive headings. Within a numbered
  question, use subheadings for its separate asks. Keep coordinate conventions,
  utility routines, stress tables, inlier timelines, estimator comparisons and
  worst-case analysis separate when discussing them. Keep each paragraph under
  the heading it actually addresses.
- Keep explanations concise without omitting reasoning. Mention every assessed
  function and explain its algorithm and important design choices, rather than
  narrating routine array operations or repeating its docstring line by line.
  Let tables carry detailed numbers and use prose to explain their significance.
- For validation choices, retain the selection criterion, a compact results
  table, the final settings and seeds, and the success-threshold rationale.
  Explain targeted tuning through the observed baseline failures and the
  parameters chosen to investigate them. Identify any changes to the default
  configuration accurately. Avoid repeating trial results or describing every
  search combination unless needed to understand the selection.
- Identify the clip, frame subset, method and processing resolution for results.
  Explain metric denominators and distinguish unavailable models from finite
  estimates outside the success tolerance. Use sufficient numerical precision
  to explain threshold decisions. Keep comparisons tied to their actual shared
  inputs and acknowledge when multiple factors change.
- Distinguish observations from interpretations. Explain the mechanism that is
  consistent with the evidence without claiming that a comparison isolates a
  cause it does not isolate. Keep ground-truth scoring and illustration separate
  from pipeline initialisation or correction.
- Include figures and tables that answer an ask or support a specific claim.
  Do not duplicate a starter notebook experiment merely to export an unnecessary
  report figure. A check required in a notebook does not automatically require
  its figure, raw output or a separate discussion in the report. Retain required
  notebook checks and evidence-table references when trimming report material.
- Keep evidence references accurate after edits. Distinguish generated output
  from added summaries and interpretation. Do not claim that results or logs
  are saved in a notebook when its code only generates or prints them, or claim
  execution or verification without supporting evidence. Remove stale references
  when removing an added figure or experiment.

### Report format and evidence references

- When revising report formatting or discussion, preserve the existing Code
  Explanation sections unless the user explicitly asks to edit them.
- When writing or explicitly revising a Code Explanation section, follow the
  report's current format: use a `### Code Explanation` heading, a `####` heading
  for each source filename in backticks, a top-level bullet for each function
  name in backticks, and nested bullets containing complete explanatory
  sentences. Describe the algorithm, important design choices and relevant
  return or failure behaviour. Keep this structured format rather than
  converting Code Explanation sections into prose paragraphs. The guidance on
  connected prose applies to the report's other explanatory sections.
- Number figures and tables by Stage, using separate sequential counters such
  as Figure 1.1 and Table 1.1 for Stage 1, and Figure 2.1 and Table 2.1 for
  Stage 2. Give every report figure and table a visible italicised caption.
  Place figure captions below their images and table captions above their
  tables. Image alt text alone is not a visible caption.
- Refer to numbered figures and tables explicitly when discussing their values
  or observations. Name the relevant panel for a multi-panel figure, such as
  "Figure 2.1 (upper left)", and identify the relevant stress-table row when
  needed. Refer to the notebook section or saved experiment log for precise
  values that are not displayed in the cited figure or table.
- Keep generated stress tables and their supplied explanatory notes unchanged.
  Put added captions and interpretation outside the generated content. Label
  added summary tables distinctly so that readers can tell them apart from
  generated output.
- Follow Stage 1's validation format: explain the observed baseline failures
  and why they motivate the chosen parameter groups, state the selection
  criterion, give a compact trial table, list the final settings and seeds,
  and explain the success tolerance and metric denominators. Identify which
  defaults changed and which parameter groups remained fixed.
- Make the purpose of each evidence subsection clear. Distinguish the PDF's
  separate evidence requirements, such as a stabilisation or failure sequence,
  from the subparts of a numbered question, such as point-level eigenvalue
  analysis. State the result first, then connect its evidence to the mechanism
  and practical implication without repeating the same explanation elsewhere.
- Clearly distinguish whole-clip results from short illustrative intervals,
  preview results from native-resolution results, and actual pipeline tracks
  from separate diagnostic probes. Keep the limitations of local examples
  explicit when they differ from the overall result.
- After adding, moving or removing figures and tables, check that numbering is
  sequential within each Stage, every prose reference matches its caption and
  evidence, image paths resolve, and evidence-table references remain accurate.
