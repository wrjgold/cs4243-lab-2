# CS4243 Lab 2

## 1. Install uv

On macOS or Linux, use the [official uv installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal, then check `uv --version`.

## 2. Set up the environment

Clone the shared repository or extract the lab release, then run these commands from its root folder:

```bash
uv venv --python 3.11
uv pip install -r code/requirements.txt ipykernel
source .venv/bin/activate
```

uv downloads Python if needed. Use Python **3.11**, the version specified by Lab 2. In VS Code, install the **Python** and **Jupyter** extensions and select `.venv/bin/python` as your Python interpreter and notebook kernel.

The environment lives in the root `.venv/`, beside `code/` and `data/`. Each collaborator creates their own environment. Do not commit or submit `.venv/`.

## 3. Prepare the data and FFmpeg

The supplied datasets are included in this repository. Keep `data/` beside `code/`; the common video is at `data/common_clip/board_clip.mp4` and stress clips are under `data/diagnostic/`.

Install FFmpeg separately. On macOS with Homebrew:

```bash
brew install ffmpeg
ffmpeg -version
```

On Linux, install FFmpeg with your system package manager. FFmpeg handles video conversion and joining generated segments.

With the environment active, extract the common clip's frames:

```bash
cd code
python -m overlaylab.extract_frames ../data/common_clip
```

## 4. Run the lab

Run lab terminal commands from `code/` with the environment activated:

```bash
python -m unittest discover -s tests -v
```

The assessed functions are implemented. The optional pyramidal-flow extension test is skipped. Open the notebooks in `code/notebooks/` using the `.venv` kernel. Notebook 4 reproduces the completed two-person Stage 4 analysis from the preserved outputs. Notebooks 1–3 also contain the original validation studies, including expensive full-resolution calibration passes. Restart a notebook's kernel before running its cells in order.

For a new terminal, activate the environment from the repository root with `source .venv/bin/activate`, then `cd code`.

See `code/README.md` for the full assignment commands and submission rules. Submit `code/` and only your created data, excluding the supplied datasets and `.venv/`.

## 5. Completed submission

The group has two required personal clips, one for Sean and one for Ruijie. Both original and corrected VACE runs are preserved and logged, and the report uses the corrected result for each person. Restart Notebook 4 and run it from top to bottom to reproduce the Scorecards, comparison tables and figures from the saved outputs. The model inputs are retained in `code/outputs/genai/<clip_id>/inputs/`, and invocation records are under `code/student_files/stage4/run_records/`.

`CS4243_Lab_2_submission.zip` contains only `code/` and the created personal data and artwork under `data/`. Supplied datasets, environments, caches, editor files and `.local_backups/` are excluded. To reproduce supplied-clip results from the submitted folders, place them beside the supplied release data.

Setup guide prepared with AI assistance. [AI-LANGUAGE]
