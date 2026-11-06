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

Some tests fail initially because the assessed functions are unfinished. Open the notebooks in `code/notebooks/` using the `.venv` kernel, and work through `01` to `04`. Restart each notebook's kernel and run its cells from top to bottom.

For a new terminal, activate the environment from the repository root with `source .venv/bin/activate`, then `cd code`.

See `code/README.md` for the full assignment commands and submission rules. Submit `code/` and only your created data, excluding the supplied datasets and `.venv/`.

Setup guide prepared with AI assistance. [AI-LANGUAGE]
