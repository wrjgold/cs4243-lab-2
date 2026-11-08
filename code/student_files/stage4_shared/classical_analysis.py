"""Complete the two personal clips' classical evaluation without retuning."""

import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from overlaylab import data, metrics as M, supplied as sp
from overlaylab.compositing import composite, occlusion_mask
from overlaylab.config import PipelineConfig
from overlaylab.evaluate import evaluate_frames
from overlaylab.geometry import dlt_homography, warp_artwork
from overlaylab.localisation import artwork_to_frame, localise_board
from overlaylab.report import mask_scores, summarise_tracking

CLIPS = {"stage4_sean": ("sean_explore", 300, 270),
         "stage4_ruijie": ("ruijie_orbit", 0, 255)}
OUT = CODE / "outputs/evaluation"
FIGURES = Path(__file__).parent


def finite(value, spec=".3f"):
    return format(value, spec) if value is not None and np.isfinite(value) else "n/a"


def percent(value):
    return f"{value:.3%}" if value is not None and np.isfinite(value) else "n/a"


def frames_at(path, indices):
    return {i: f for i, f in sp.iter_video(path, stop=max(indices)+1) if i in set(indices)}


def saved_h(stats, frame, reference):
    q = np.asarray(stats["corners"][frame], float)
    if stats["status"][frame] != "ok" or stats["flags"][frame][0] == "HIDE" or not np.isfinite(q).all():
        return None
    H, _, ok = dlt_homography(sp.rect_corners(reference.shape[1], reference.shape[0]), q)
    return H if ok else None


def episodes(flags):
    result, start = [], None
    for i, (flag, _) in enumerate(flags + [["END", "none"]]):
        if flag == "HIDE" and start is None:
            start = i
        elif flag != "HIDE" and start is not None:
            result.append([start, i-1])
            start = None
    return result


def match_picture(reference, frame, diag, matches):
    height = 240
    a = cv2.resize(reference, (round(reference.shape[1]*height/reference.shape[0]), height))
    b = cv2.resize(frame, (round(frame.shape[1]*height/frame.shape[0]), height))
    image = np.concatenate([a, b], axis=1).copy()
    for ri, fi in matches[:30]:
        x = diag["keypoints_ref"][ri] * height/reference.shape[0]
        y = diag["keypoints_frame"][fi] * height/frame.shape[0] + [a.shape[1], 0]
        cv2.line(image, tuple(np.round(x).astype(int)), tuple(np.round(y).astype(int)), (1., .25, .1), 1, cv2.LINE_AA)
    return image


def make_figures(clip, stats, gt, sources, overlays, reference, cfg, representative, failure):
    flow = stats["flow_assisted"]
    H = saved_h(flow, representative, reference)
    difference = np.zeros(sources[representative].shape[:2], np.float32)
    predicted = np.zeros_like(difference, bool)
    if H is not None:
        predicted, difference, _ = occlusion_mask(sources[representative], H, reference, cfg.occlusion)
    _, _, pairs, _, diag = localise_board(reference, sources[failure], cfg.localisation)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, frame in enumerate([representative, failure]):
        axes[row, 0].imshow(sources[frame]); axes[row, 0].set_title(f"Source frame {frame}")
        axes[row, 1].imshow(overlays[frame]); axes[row, 1].set_title(f"Flow-assisted: {flow['flags'][frame][0]}")
        for col in [0, 1]:
            q = gt[frame]
            axes[row, col].plot(*np.vstack([q, q[0]]).T, color="lime", linewidth=1)
    axes[0, 2].imshow(difference, cmap="inferno", vmin=0, vmax=.5)
    if predicted.any():
        axes[0, 2].contour(predicted, levels=[.5], colors="cyan", linewidths=.7)
    axes[0, 2].set_title("Compensated difference / cyan predicted mask")
    axes[1, 2].imshow(match_picture(reference, sources[failure], diag, pairs))
    axes[1, 2].set_title(f"Failed fixed localiser: {diag['n_matches']} matches, {diag['n_inliers']} inliers")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{clip}: full-resolution source and frozen classical evidence")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{clip}_classical_evidence.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    colors = {"independent":"tab:blue", "flow_assisted":"tab:orange"}
    for row, (variant, st) in enumerate(stats.items()):
        x = np.arange(st["n_frames"])
        axes[0].plot(x, [s["inlier_count"] for s in st["stats"]], color=colors[variant], label=variant, alpha=.8)
        errors = [(f, M.corner_error(np.asarray(st["corners"][f]), q)*100) for f, q in gt.items()
                  if st["status"][f] == "ok" and st["flags"][f][0] != "HIDE"]
        axes[1].scatter([a for a,b in errors], [b for a,b in errors], color=colors[variant], s=25, label=variant)
        flags = np.array([0 if z[0] == "OK" else 1 if z[0] == "REINIT" else 2 for z in st["flags"]])
        axes[2].imshow(flags[None], aspect="auto", extent=[-.5,449.5,row-.4,row+.4],
                       cmap=plt.matplotlib.colors.ListedColormap(["#55aa66","#e5b54d","#cc6677"]), vmin=0, vmax=2)
    axes[0].axhline(cfg.track.monitor.min_inlier_count, color="black", linestyle="--", linewidth=.8)
    axes[0].set_ylabel("Inliers"); axes[0].legend(loc="upper right")
    axes[1].axhline(cfg.localisation.success_threshold*100, color="black", linestyle="--", linewidth=.8)
    axes[1].set_ylabel("Finite audit error (%)"); axes[1].legend(loc="upper right")
    axes[2].set_yticks([0,1], ["independent","flow_assisted"])
    axes[2].set_xlabel("Prepared source frame index (green OK / yellow REINIT / red HIDE)")
    for ax in axes:
        a, b = (233,313) if clip == "stage4_sean" else (340,395)
        ax.axvspan(a, b, color="skyblue", alpha=.15)
        ax.axvline(representative, color="grey", linestyle=":", alpha=.7)
        ax.axvline(failure, color="purple", linestyle=":", alpha=.7)
        ax.grid(alpha=.15)
    fig.suptitle(f"{clip}: frozen monitor and available audit predictions (unavailable errors omitted)")
    fig.tight_layout(); fig.savefig(FIGURES / f"{clip}_classical_timeline.png", dpi=140); plt.close(fig)
    truth = data.load_audit_masks(clip).get(representative)
    return {"failure_frame":failure,"representative_frame":representative,
            "failure_localiser":{"matches":diag["n_matches"],"inliers":diag["n_inliers"],
                                 "status":bool(diag["status"]),"median_residual":diag["median_residual"]},
            "representative_mean_difference_inside_board":float(difference[predicted].mean()) if predicted.any() else None,
            "strong_fraction_of_gt_occluder":float((difference[truth] >= cfg.occlusion.strong).mean()) if truth is not None and truth.any() else None,
            "predicted_fraction_of_gt_occluder":float(predicted[truth].mean()) if truth is not None and truth.any() else None}


def analyse():
    cfg = PipelineConfig.from_json((CODE / "outputs/thresholds.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    score_dir = CODE / "outputs/scorecards"; score_dir.mkdir(parents=True, exist_ok=True)
    captures = {r["clip_id"]:r for r in sp.read_csv(CODE / "manifests/capture_manifest.csv")}
    all_results = {}
    for clip, (art_name, representative, failure) in CLIPS.items():
        video, ref_path = data.clip_paths(clip)
        reference, art = sp.read_image(ref_path), data.load_artwork(art_name)
        cfg.localisation.warp.reference_shape = tuple(reference.shape[:2])
        gt, masks = data.load_audit_corners(clip), data.load_audit_masks(clip)
        needed = sorted({0,representative,failure,*masks})
        source = frames_at(video, needed)
        _, _, _, _, init = localise_board(reference, source[0], cfg.localisation)
        init_record = {k:init[k] for k in ["n_keypoints_ref","n_keypoints_frame","n_matches","n_inliers","inlier_ratio","median_residual"]}
        result = {"clip_id":clip,"processing_resolution":[1280,720],"audit_frames":sorted(gt),
                  "mask_frames":sorted(masks),"initialisation":init_record,"variants":{},"capture":captures[clip]}
        stats = {}
        for variant in ["independent","flow_assisted"]:
            path = CODE / f"outputs/videos/{clip}_{variant}_stats.json"
            st = json.loads(path.read_text()); stats[variant] = st
            output = path.with_name(f"{clip}_{variant}.mp4")
            assert sp.video_info(output) == {"frames":450,"fps":30.,"width":1280,"height":720}
            assert st["n_frames"] == 450 and st["resize_width"] is None
            expected = copy.deepcopy(cfg); expected.track.variant = variant
            assert st["frozen_config"] == expected.thresholds()
            assert st["request"]["source_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
            assert st["request"]["artwork_sha256"] == hashlib.sha256(np.ascontiguousarray(art).tobytes()).hexdigest()
            track = summarise_tracking(st, gt, cfg.localisation.success_threshold, 1280)
            mask_stats = copy.deepcopy(st)
            mask_stats["mask_dir"] = str(path.with_name(f"{clip}_{variant}_masks"))
            score = mask_scores(mask_stats, masks, sorted(masks))
            alpha_with, alpha_without, per_mask = [], [], []
            for f, truth in masks.items():
                if not truth.any():
                    continue
                H = saved_h(st, f, reference)
                masked_alpha = baseline_alpha = np.zeros(truth.shape, np.float32)
                if H is not None:
                    pred, _, _ = occlusion_mask(source[f], H, reference, cfg.occlusion)
                    Ha = artwork_to_frame(art, H, cfg.localisation)
                    warped, coverage = warp_artwork(art, Ha, source[f].shape)
                    _, masked_alpha = composite(source[f], warped, coverage, pred, cfg.composite)
                    _, baseline_alpha = composite(source[f], warped, coverage, np.zeros(truth.shape,bool), cfg.composite)
                with_mask, without_mask = M.leakage(masked_alpha,truth), M.leakage(baseline_alpha,truth)
                alpha_with.append(with_mask); alpha_without.append(without_mask)
                per_mask.append({"frame":f,"hidden":H is None,"alpha_leakage":with_mask,"no_occlusion_alpha_leakage":without_mask})
            rendered = json.loads((OUT / f"{clip}_{variant}_report.json").read_text())
            conditions = json.loads((CODE / f"outputs/genai/{clip}/conditions.json").read_text())
            comparison = conditions["frames"]
            chosen = sorted({comparison[k] for k in ["clean","occluded","perspective"]} | set(comparison["pair"]))
            frame_dir = CODE / f"outputs/genai/{clip}/classical_{variant}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for f, image in frames_at(output, chosen).items():
                sp.write_image(frame_dir / f"frame_{f:06d}.png", image)
            baseline_report = evaluate_frames([frame_dir / f"frame_{f:06d}.png" for f in chosen],
                video, art, gt_corners=gt, gt_masks=masks,
                success_threshold=cfg.localisation.success_threshold, verbose=False)
            (OUT / f"{clip}_{variant}_frames_report.json").write_text(json.dumps(baseline_report,indent=2)+"\n")
            result["variants"][variant] = {"tracking":track,"mask_scores":score,"rendered":rendered,
                "time_s":sum(st["timing"].values()),"tracking_s":st["timing"]["track_s"],
                "composite_s":st["timing"]["composite_s"],"hide_episodes":episodes(st["flags"]),
                "triggers":dict(Counter(x[1] for x in st["flags"])),
                "flow_frames":sum(bool(z) for z in st["flow"]),
                "comparison_baseline":baseline_report,
                "direct_alpha_leakage":float(np.mean(alpha_with)),
                "baseline_alpha_leakage":float(np.mean(alpha_without)),"alpha_mask_frames":per_mask}
        overlay = frames_at(CODE / f"outputs/videos/{clip}_flow_assisted.mp4", [representative,failure])
        result["evidence"] = make_figures(clip, stats, gt, source, overlay, reference, cfg, representative, failure)
        q = gt[0];result["first_frame_board_fraction"] = float(abs(np.dot(q[:,0],np.roll(q[:,1],-1))-np.dot(q[:,1],np.roll(q[:,0],-1)))/2/(1280*720))
        fields = {"Clip":f"{clip}, 15 s, 1280 x 720 at 30 fps. {captures[clip]['board_description']}. First-frame annotated area {result['first_frame_board_fraction']:.2%}. Event ranges: {captures[clip]['event_frame_ranges']}.",
          "Initialisation":f"Descriptor-valid keypoints reference/frame 0: {init['n_keypoints_ref']}/{init['n_keypoints_frame']}. Matches {init['n_matches']}, inliers {init['n_inliers']} ({init['inlier_ratio']:.3f}), median residual {init['median_residual']:.3f} px.",
          "Geometry":"", "Temporal":"", "Occlusion":"", "Monitor":"",
          "Description":("The overlay is available on only four of ten audit frames. Frame 300 paints much of the purple foreground object, while frame 270 is hidden after localisation failure." if clip=='stage4_sean' else "The overlay is available on three of ten audit frames. The strongest-perspective frame and all five mask frames are hidden, so zero leakage is explained by absent overlays."),
          "Uncertainty":"All settings and seed 0 were transferred without personal-clip tuning. Dimensions and event ranges are approximate. Manual corners behind occluders are uncertain. Tracker jitter uses sparse GT motion interpolation. Rendered placement uses SIFT re-localisation and differs from saved-corner error. No full board exit/re-entry occurs, so re-entry recovery is not assessed."}
        for variant, v in result['variants'].items():
            t, m, e = v['tracking'],v['mask_scores'],v['rendered']
            fields['Geometry'] += f"{variant}: median saved-corner error {t['median_error']:.3%} on {t['n']-t['failed']}/10 finite frames, success {t['success']:.0%} at 2%, unavailable {t['failed']}/10. "
            fields['Temporal'] += f"{variant}: tracker jitter {finite(t['jitter_px'])} px, drift at least 16 frames after reinitialisation {percent(t['drift_error'])} on {t['drift_frames']} finite audit frames, longest localisation interval {t['longest_run']} frames. "
            fields['Occlusion'] += f"{variant}: IoU {finite(m['iou'])}, F1 {finite(m['f1'])} over {m['mask_frames_scored']}/3 nonempty masks, hidden {len(m['mask_frames_hidden'])}, missing {len(m['mask_frames_missing'])}. Finished-video leakage {e['occlusion']['leakage']:.2%} over all 3 nonempty masks. Coverage {percent(e['occlusion']['coverage_visible'])} on {e['occlusion']['n_coverage_frames']} overlay-present mask frames. Direct alpha leakage with/without masking {v['direct_alpha_leakage']:.2%}/{v['baseline_alpha_leakage']:.2%}, same saved geometry and HIDE policy. "
            fields['Monitor'] += f"{variant}: {t['hide']}/450 HIDE frames, {len(v['hide_episodes'])} HIDE episodes, {t['reinit']} REINIT flags, {t['localisations']} localisation calls, {t['frames_tracked']}/450 tracked. Pipeline time {v['time_s']:.2f} s. "
        text = '\n'.join([f"# Overlay Scorecard — {clip}",'']+[f"**{k}**: {v.rstrip()}\n" for k,v in fields.items()])
        text += f"\nEvidence and monitor: [source/overlay/failure](../../student_files/stage4_shared/{clip}_classical_evidence.png), [timeline](../../student_files/stage4_shared/{clip}_classical_timeline.png).\n"
        (score_dir / f"{clip}_scorecard.md").write_text(text.rstrip()+'\n')
        all_results[clip] = result
        print(clip, 'classical analysis and Scorecard saved.')
    (OUT / 'stage4_classical_summary.json').write_text(json.dumps(all_results,indent=2)+'\n')
    return all_results


if __name__ == '__main__':
    analyse()
