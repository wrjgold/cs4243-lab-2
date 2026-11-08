"""Build a per-person VACE input ZIP and import its unmodified Colab results.

Run bundle only after manual annotation and the final frozen pipeline run.
This script does not run the classical pipeline or any generative model.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import cv2
import numpy as np
from PIL import Image

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from overlaylab import data, supplied as sp
from overlaylab.annotate import check
from overlaylab.config import PipelineConfig

MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
REVISION = "ec4d2cb062b548996b179d493fdd05340de702a1"
CLIPS = ("stage4_sean", "stage4_ruijie")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def person_dir(clip):
    return CODE / "outputs" / "genai" / clip


def bundle(clip):
    folder = person_dir(clip)
    conditions = json.loads((folder / "conditions.json").read_text())
    video, reference = data.clip_paths(clip)
    artwork = (CODE / "manifests" / conditions["artwork"]).resolve()
    stats_path = CODE / "outputs" / "videos" / f"{clip}_flow_assisted_stats.json"
    if not stats_path.exists():
        raise SystemExit(f"Not ready: {stats_path.name} is missing. Finish annotations and run the final frozen pipeline first.")
    if check(clip):
        raise SystemExit("Fix audit annotation problems before preparing model inputs.")
    stats = json.loads(stats_path.read_text())
    info = sp.video_info(video)
    expected = PipelineConfig.from_json((CODE / "outputs/thresholds.json").read_text())
    expected.track.variant = "flow_assisted"
    request = stats.get("request", {})
    fingerprint = hashlib.sha256()
    for path in sorted((CODE / "overlaylab").glob("*.py")):
        fingerprint.update(path.name.encode())
        fingerprint.update(path.read_bytes())
    if request.get("code_sha256") != fingerprint.hexdigest():
        raise SystemExit("Pipeline statistics were produced by different code. Use the final frozen pipeline results.")
    if stats.get("n_frames") != info["frames"] or len(stats["corners"]) != info["frames"]:
        raise SystemExit("The pipeline statistics must cover the complete prepared clip.")
    if request.get("source_sha256") != sha256(video) or request.get("config") != expected.thresholds():
        raise SystemExit("Pipeline source/settings do not match the prepared clip and current frozen thresholds.")
    expected_reference = sha256_array(sp.read_image(reference))
    if request.get("reference_sha256") != expected_reference:
        raise SystemExit("Pipeline reference differs from the current rectified reference.")
    frames = conditions["frames"]
    audit_c = data.load_audit_corners(clip)
    audit_m = data.load_audit_masks(clip)
    named = [frames[k] for k in ("clean", "occluded", "perspective")]
    if len(set(named)) != 3 or not set(named) <= set(audit_c) or frames["occluded"] not in audit_m:
        raise SystemExit("The three comparison frames must be distinct audit-corner frames, with a mask on the occluded one.")
    if frames["pair"][1] != frames["pair"][0] + 1:
        raise SystemExit("The temporal pair must be consecutive.")
    start, end = (conditions["model"]["interval"][k] for k in ("start", "end"))
    chosen = sorted(set(named + frames["pair"]))
    settings = conditions["model"]["settings"]
    output_fps = conditions["model"]["output_fps"]
    assert 0 <= start <= min(chosen) <= max(chosen) <= end < info["frames"]
    sample_indices = [start + round(k * info["fps"] / output_fps) for k in range(settings["num_frames"])]
    if sample_indices[-1] != end or len(set(round((f-start)*output_fps/info["fps"]) for f in frames["pair"])) != 2:
        raise SystemExit("Output duration or temporal-pair mapping is inconsistent with the fixed run.")
    archive = folder / f"{clip}_inputs.zip"
    if archive.exists():
        raise SystemExit(f"Input ZIP already exists: {archive}. Preserve fixed inputs after generation.")
    with tempfile.TemporaryDirectory(prefix="stage4-inputs-") as tmp:
        stage = Path(tmp)
        (stage / "masks").mkdir()
        shutil.copyfile(artwork, stage / "artwork.png")
        count, nonempty = 0, 0
        with sp.VideoWriter(stage / "interval.mp4", info["fps"], (info["width"], info["height"])) as writer:
            for i, frame in sp.iter_video(video, start=start, stop=end+1):
                writer.write(frame)
                q = np.asarray(stats["corners"][i], dtype=float)
                q = q * info["width"] / (stats.get("resize_width") or info["width"])
                valid = q.shape == (4, 2) and np.isfinite(q).all()
                mask = sp.quad_mask(frame.shape[:2], q) if valid else np.zeros(frame.shape[:2], bool)
                sp.write_mask(stage / "masks" / f"mask_{i:06d}.png", mask)
                nonempty += bool(mask.any())
                count += 1
        assert count == end-start+1
        assets = {p.relative_to(stage).as_posix(): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        run = {
            "schema": "cs4243-vace-inputs-v1", "clip_id": clip, "conditions": conditions,
            "model_id": MODEL_ID, "model_revision": REVISION,
            "source": {**info, "sha256": sha256(video), "interval_start": start,
                       "interval_end": end, "interval_frames": count},
            "sample_source_indices": sample_indices,
            "mask_origin": "quadrilaterals from the final flow_assisted pipeline, not audit masks",
            "pipeline_stats_sha256": sha256(stats_path),
            "thresholds_sha256": sha256(CODE / "outputs/thresholds.json"),
            "artwork_sha256": sha256(artwork), "assets_sha256": assets,
            "input_codec": "interval.mp4 written by the supplied VideoWriter, matching Notebook 4",
            "nonempty_board_masks": nonempty,
        }
        (stage / "run.json").write_text(json.dumps(run, indent=2)+"\n")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(stage).as_posix())
    print(f"Ready to upload in Colab: {archive}")
    print(f"{count} source/mask frames, {nonempty} nonempty board masks. No audit annotation is included in the ZIP.")


def sha256_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def extract_frames(video, indices, destination, original_size, output_fps=None, source_fps=30, start=0):
    destination.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    try:
        for f in indices:
            k = round((f-start)*output_fps/source_fps) if output_fps else f
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot extract output index {k} for source frame {f}.")
            if (frame.shape[1], frame.shape[0]) != original_size:
                frame = cv2.resize(frame, original_size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(destination / f"frame_{f:06d}.png"), frame)
    finally:
        cap.release()


def import_results(archive):
    with zipfile.ZipFile(archive) as z:
        roots = {Path(n).parts[0] for n in z.namelist() if Path(n).parts}
        if len(roots) != 1 or next(iter(roots)) not in CLIPS:
            raise SystemExit("Expected one person's results ZIP from the supplied Colab notebook.")
        clip = next(iter(roots))
        run = json.loads(z.read(f"{clip}/run.json"))
        video, _ = data.clip_paths(clip)
        if run["source"]["sha256"] != sha256(video):
            raise SystemExit("Results belong to a different prepared source video.")
        current = json.loads((person_dir(clip) / "conditions.json").read_text())
        if run["conditions"] != current:
            raise SystemExit("Results do not match this person's fixed conditions.")
        destination = person_dir(clip)
        for name in z.namelist():
            if name.endswith("/"):
                continue
            relative = Path(name).relative_to(clip)
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise SystemExit("Invalid result ZIP member path.")
            contents = z.read(name)
            if target.exists() and target.read_bytes() != contents and relative.as_posix() == "genai_log.csv":
                import io
                with target.open(newline="") as f:
                    previous_rows = list(csv.DictReader(f))
                incoming_rows = list(csv.DictReader(io.StringIO(contents.decode())))
                if incoming_rows[:len(previous_rows)] == previous_rows:
                    continue
            if target.exists() and target.read_bytes() != contents:
                raise SystemExit(f"Refusing to overwrite different existing results: {target}")
        for name in z.namelist():
            if name.endswith("/"):
                continue
            target = destination / Path(name).relative_to(clip)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
    log = CODE / "manifests/genai_log.csv"
    with (CODE / "manifests/genai_log_template.csv").open(newline="") as f:
        fields = next(csv.reader(f))
    with log.open(newline="") if log.exists() else open(CODE / "manifests/genai_log_template.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    with (destination / "genai_log.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row not in rows:
                rows.append(row)
    with log.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    generated = destination / "output.mp4"
    if not generated.exists():
        print(f"Imported technical failure records for {clip}. No successful video was present.")
        return
    c = run["conditions"]
    chosen = sorted(set([c["frames"][k] for k in ("clean", "occluded", "perspective")] + c["frames"]["pair"]))
    size = (run["source"]["width"], run["source"]["height"])
    extract_frames(generated, chosen, destination, size, c["model"]["output_fps"],
                   run["source"]["fps"], run["source"]["interval_start"])
    for variant in ("independent", "flow_assisted"):
        classical = CODE / "outputs/videos" / f"{clip}_{variant}.mp4"
        if classical.exists():
            extract_frames(classical, chosen, destination / f"classical_{variant}", size)
    print(f"Imported unchanged raw output and extracted nominal-time comparison frames: {destination}")
    print("Evaluate these PNGs with overlaylab.evaluate --frames, using the same --clip and artwork for both methods.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("clip", choices=CLIPS)
    i = sub.add_parser("import-results")
    i.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.command == "bundle":
        bundle(args.clip)
    else:
        import_results(args.archive)


if __name__ == "__main__":
    main()
