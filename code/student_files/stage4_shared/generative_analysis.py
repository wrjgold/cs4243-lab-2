"""Import both recorded attempts per student and reproduce the Stage 4 comparison."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).parent))
from overlaylab import data, supplied as sp
from overlaylab.config import PipelineConfig
from overlaylab.evaluate import evaluate_frames
from handoff import extract_frames

HERE = Path(__file__).parent
RECORDS = HERE / "run_records"
CLIPS = {"stage4_sean": "sean_explore", "stage4_ruijie": "ruijie_orbit"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float) + "\n")


def import_archives(paths):
    entries, rows = [], []
    for path in paths:
        path = Path(path)
        with zipfile.ZipFile(path) as archive:
            roots = {Path(n).parts[0] for n in archive.namelist() if Path(n).parts}
            assert len(roots) == 1 and next(iter(roots)) in CLIPS
            clip = next(iter(roots))
            names = [n for n in archive.namelist() if n.endswith("/run_metadata.json")]
            assert len(names) == 1
            metadata = json.loads(archive.read(names[0]))
            conditions = json.loads(archive.read(f"{clip}/conditions.json"))
            run = json.loads(archive.read(f"{clip}/run.json"))
            invocation = metadata["invocation_id"]
            video, _ = data.clip_paths(clip)
            assert run["source"]["sha256"] == digest(video) == metadata["source_sha256"]
            assert metadata["status"] == "completed"
            assert metadata["conditions"] == conditions
            original_conditions = json.loads(json.dumps(run["conditions"]))
            original_conditions["prompt"] = conditions["prompt"]
            assert original_conditions == conditions
            assert conditions["model"]["seed"] == 0
            assert conditions["model"]["settings"] == run["conditions"]["model"]["settings"]
            raw = archive.read(f"{clip}/output.mp4")
            assert hashlib.sha256(raw).hexdigest() == metadata["output_sha256"]
            assert raw == archive.read(f"{clip}/invocations/{invocation}/output.mp4")
            destination = CODE / "outputs/genai" / clip / "invocations" / invocation
            destination.mkdir(parents=True, exist_ok=True)
            output = destination / "output.mp4"
            if output.exists():
                assert output.read_bytes() == raw
            else:
                output.write_bytes(raw)
            record_dir = RECORDS / clip / invocation
            record_dir.mkdir(parents=True, exist_ok=True)
            for name in ("conditions.json", "genai_log.csv", "input_check.json", "run.json"):
                (record_dir / name).write_bytes(archive.read(f"{clip}/{name}"))
            (record_dir / "run_metadata.json").write_bytes(archive.read(names[0]))
            with (record_dir / "genai_log.csv").open(newline="") as stream:
                invocation_rows = list(csv.DictReader(stream))
            assert len(invocation_rows) == 1
            assert invocation_rows[0]["prompt"] == conditions["prompt"]["template"]
            rows.extend(invocation_rows)
            role = "corrected" if "prompt_revision" in path.stem else "original"
            entries.append({"clip_id": clip, "role": role, "invocation_id": invocation,
                            "archive_name": path.name, "archive_sha256": digest(path),
                            "records": str(record_dir.relative_to(CODE)),
                            "output": str(output.relative_to(CODE)),
                            "output_sha256": metadata["output_sha256"]})
    assert {(e["clip_id"], e["role"]) for e in entries} == {
        (c, r) for c in CLIPS for r in ("original", "corrected")}
    entries.sort(key=lambda e: e["invocation_id"])
    write_json(RECORDS / "index.json", entries)
    with (CODE / "manifests/genai_log_template.csv").open(newline="") as stream:
        fields = next(csv.reader(stream))
    with (CODE / "manifests/genai_log.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["date"]))
    for clip in CLIPS:
        selected = next(e for e in entries if e["clip_id"] == clip and e["role"] == "corrected")
        conditions = json.loads((CODE / selected["records"] / "conditions.json").read_text())
        folder = CODE / "outputs/genai" / clip
        write_json(folder / "conditions.json", conditions)
        shutil.copyfile(CODE / selected["output"], folder / "output.mp4")
    publish_active()
    return entries


def publish_active():
    folder = CODE / "outputs/genai/stage4_sean"
    shutil.copyfile(folder / "conditions.json", CODE / "manifests/genai_conditions.json")
    shutil.copyfile(folder / "output.mp4", CODE / "outputs/genai/output.mp4")
    if (folder / "inputs").exists():
        shutil.copytree(folder / "inputs", CODE / "outputs/genai/inputs", dirs_exist_ok=True)
    for p in folder.glob("frame_*.png"):
        shutil.copyfile(p, CODE / "outputs/genai" / p.name)
    for variant in ("independent", "flow_assisted"):
        source = folder / f"classical_{variant}"
        if source.exists():
            shutil.copytree(source, CODE / "outputs/genai" / source.name, dirs_exist_ok=True)


def export_inputs(clip):
    folder = CODE / "outputs/genai" / clip
    destination = folder / "inputs"
    archive = HERE / "input_bundles" / f"{clip}_inputs.zip"
    if not archive.exists():
        archive = folder / f"{clip}_inputs.zip"
    with zipfile.ZipFile(archive) as source:
        run = json.loads(source.read("run.json"))
        assert run["thresholds_sha256"] == digest(CODE / "outputs/thresholds.json")
        assert run["pipeline_stats_sha256"] == digest(CODE / "outputs/videos" / f"{clip}_flow_assisted_stats.json")
        for name, expected in run["assets_sha256"].items():
            raw = source.read(name)
            assert hashlib.sha256(raw).hexdigest() == expected
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    return run


def source_baseline(clip, run, chosen):
    video, _ = data.clip_paths(clip)
    folder = HERE / "source_baseline" / clip
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "reencoded.mp4"
    wanted = set(run["sample_source_indices"])
    with sp.VideoWriter(path, 16, (832, 480)) as writer:
        for index, frame in sp.iter_video(video, start=min(wanted), stop=max(wanted) + 1):
            if index in wanted:
                writer.write(cv2.resize(frame, (832, 480), interpolation=cv2.INTER_LANCZOS4))
    extract_frames(path, chosen, folder, (1280, 720), 16, 30, run["source"]["interval_start"])
    return folder


def comparison_figure(clip, conditions, original_folder, corrected_folder):
    video, _ = data.clip_paths(clip)
    gt = data.load_audit_corners(clip)
    named = [(label, conditions["frames"][label]) for label in ("clean", "occluded", "perspective")]
    source = {i: f for i, f in sp.iter_video(video, stop=max(f for _, f in named) + 1)
              if i in {f for _, f in named}}
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for row, (label, index) in enumerate(named):
        paths = [None, CODE / "outputs/genai" / clip / "classical_flow_assisted" / f"frame_{index:06d}.png",
                 original_folder / f"frame_{index:06d}.png", corrected_folder / f"frame_{index:06d}.png"]
        for col, path in enumerate(paths):
            image = source[index] if path is None else sp.read_image(path)
            axes[row, col].imshow(image)
            axes[row, col].set_title(f"{label}, source {index}" if col == 0 else
                                     ("Classical flow-assisted", "Original VACE", "Corrected VACE")[col - 1])
            q = gt[index]
            axes[row, col].plot(*np.vstack([q, q[0]]).T, color="lime", linewidth=.6)
            axes[row, col].set_xlim(-.5, 1279.5)
            axes[row, col].set_ylim(719.5, -.5)
            axes[row, col].axis("off")
    fig.suptitle(f"{clip}: fixed audit frames at nominal timestamps; green is the source audit quadrilateral")
    fig.subplots_adjust(left=.015, right=.985, bottom=.025, top=.91, wspace=.045, hspace=.24)
    fig.savefig(HERE / f"{clip}_generative_comparison.png", dpi=160)
    plt.close(fig)


def fmt(value, pattern=".3f"):
    return format(value, pattern) if value is not None and np.isfinite(value) else "n/a"


def comparison_markdown(results):
    lines = ["| Clip / method | Detected | Median error | NCC | Background PSNR | Leakage | Coverage (n) | Flicker | Warp error | Time (s) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for clip, result in results.items():
        for method in ("classical_independent", "classical_flow_assisted", "generative_original", "generative"):
            report = result["reports"][method]
            p, f, s, o, t = [report[k] for k in ("placement", "fidelity", "preservation", "occlusion", "temporal")]
            lines.append(f"| {clip} / {method} | {p['n']-p['failed']}/{p['n']} | {fmt(p['median'], '.2%')} | "
                         f"{fmt(f['ncc'])} | {fmt(s['psnr_outside'], '.2f')} | {fmt(o['leakage'], '.2%')} | "
                         f"{fmt(o['coverage_visible'], '.2%')} ({o['n_coverage_frames']}) | {fmt(t['flicker'], '.4f')} | "
                         f"{fmt(t['warping_error'], '.4f')} | {fmt(result['costs'][method]['seconds'], '.2f')} |")
    return "\n".join(lines)


def analyse():
    entries = json.loads((RECORDS / "index.json").read_text())
    frozen = PipelineConfig.from_json((CODE / "outputs/thresholds.json").read_text())
    results = {}
    evaluation = CODE / "outputs/evaluation"
    for clip, art_name in CLIPS.items():
        run = export_inputs(clip)
        folder = CODE / "outputs/genai" / clip
        conditions = json.loads((folder / "conditions.json").read_text())
        fr = conditions["frames"]
        chosen = sorted({fr[k] for k in ("clean", "occluded", "perspective")} | set(fr["pair"]))
        mapping = {f: round((f-run['source']['interval_start'])*16/30) for f in chosen}
        assert len(set(mapping.values())) == len(chosen)
        video, _ = data.clip_paths(clip)
        gt, masks, artwork = data.load_audit_corners(clip), data.load_audit_masks(clip), data.load_artwork(art_name)
        reports, costs = {}, {}
        for role in ("original", "corrected"):
            entry = next(e for e in entries if e["clip_id"] == clip and e["role"] == role)
            raw = CODE / entry["output"]
            assert digest(raw) == entry["output_sha256"]
            assert sp.video_info(raw) == {"frames": 81, "fps": 16., "width": 832, "height": 480}
            target = raw.parent
            extract_frames(raw, chosen, target, (1280, 720), 16, 30, run["source"]["interval_start"])
            method = "generative" if role == "corrected" else "generative_original"
            reports[method] = evaluate_frames([target / f"frame_{f:06d}.png" for f in chosen], video, artwork,
                gt_corners=gt, gt_masks=masks, success_threshold=frozen.localisation.success_threshold, verbose=False)
            suffix = "genai" if role == "corrected" else "genai_original"
            write_json(evaluation / f"{clip}_{suffix}_frames_report.json", reports[method])
            meta = json.loads((CODE / entry["records"] / "run_metadata.json").read_text())
            costs[method] = {"seconds": meta["elapsed_s"], "setup_seconds": meta["setup"]["model_setup_s"],
                             "compute": f"Google Colab {meta['setup']['gpu']}", "invocation_id": entry["invocation_id"]}
            if role == "corrected":
                for path in target.glob("frame_*.png"):
                    shutil.copyfile(path, folder / path.name)
                corrected_folder = target
            else:
                original_folder = target
        for variant in ("independent", "flow_assisted"):
            target = folder / f"classical_{variant}"
            extract_frames(CODE / "outputs/videos" / f"{clip}_{variant}.mp4", chosen, target, (1280, 720))
            method = f"classical_{variant}"
            reports[method] = evaluate_frames([target / f"frame_{f:06d}.png" for f in chosen], video, artwork,
                gt_corners=gt, gt_masks=masks, success_threshold=frozen.localisation.success_threshold, verbose=False)
            write_json(evaluation / f"{clip}_{variant}_frames_report.json", reports[method])
            stats = json.loads((CODE / "outputs/videos" / f"{clip}_{variant}_stats.json").read_text())
            costs[method] = {"seconds": sum(stats["timing"].values()), "compute": "Local CPU, hardware not recorded",
                             "source_frames": stats["n_frames"]}
        target = source_baseline(clip, run, chosen)
        reports["source_reencoded"] = evaluate_frames([target / f"frame_{f:06d}.png" for f in chosen], video, artwork,
            gt_corners=gt, gt_masks=masks, success_threshold=frozen.localisation.success_threshold, verbose=False)
        write_json(evaluation / f"{clip}_source_reencoded_frames_report.json", reports["source_reencoded"])
        results[clip] = {"conditions": conditions, "source_indices": chosen, "output_indices": mapping,
                         "reports": reports, "costs": costs, "attempts": [e for e in entries if e["clip_id"] == clip]}
        comparison_figure(clip, conditions, original_folder, corrected_folder)
    write_json(evaluation / "stage4_genai_reports.json", results)
    write_json(evaluation / "genai_frames_report.json", results["stage4_sean"]["reports"]["generative"])
    for variant in ("independent", "flow_assisted"):
        write_json(evaluation / f"classical_{variant}_frames_report.json", results["stage4_sean"]["reports"][f"classical_{variant}"])
    publish_active()
    (HERE / "generative_comparison.md").write_text(comparison_markdown(results) + "\n")
    print(comparison_markdown(results))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-results", nargs=4, type=Path)
    args = parser.parse_args()
    if args.import_results:
        import_archives(args.import_results)
    analyse()
