# Overlay Scorecard — stage4_ruijie

**Clip**: stage4_ruijie, 15 s, 1280 x 720 at 30 fps. Printed grey abstract artwork on the flat rectangular front face of a box, including its outer white border and excluding the side panel. First-frame annotated area 44.16%. Event ranges: {"first_viewing_angle_change_approx":[90,150],"strongest_side_view_approx":[225,285],"occlusion_crossing_approx":[340,395]}.

**Initialisation**: Descriptor-valid keypoints reference/frame 0: 60/150. Matches 45, inliers 24 (0.533), median residual 0.479 px.

**Geometry**: independent: median saved-corner error 0.381% on 3/10 finite frames, success 30% at 2%, unavailable 7/10. flow_assisted: median saved-corner error 0.381% on 3/10 finite frames, success 30% at 2%, unavailable 7/10.

**Temporal**: independent: tracker jitter 4.969 px, drift at least 16 frames after reinitialisation n/a on 0 finite audit frames, longest localisation interval 1 frames. flow_assisted: tracker jitter 3.908 px, drift at least 16 frames after reinitialisation 0.414% on 1 finite audit frames, longest localisation interval 23 frames.

**Occlusion**: independent: IoU n/a, F1 n/a over 0/3 nonempty masks, hidden 3, missing 0. Finished-video leakage 0.00% over all 3 nonempty masks. Coverage n/a on 0 overlay-present mask frames. Direct alpha leakage with/without masking 0.00%/0.00%, same saved geometry and HIDE policy. flow_assisted: IoU n/a, F1 n/a over 0/3 nonempty masks, hidden 3, missing 0. Finished-video leakage 0.00% over all 3 nonempty masks. Coverage n/a on 0 overlay-present mask frames. Direct alpha leakage with/without masking 0.00%/0.00%, same saved geometry and HIDE policy.

**Monitor**: independent: 280/450 HIDE frames, 31 HIDE episodes, 0 REINIT flags, 450 localisation calls, 170/450 tracked. Pipeline time 129.88 s. flow_assisted: 275/450 HIDE frames, 27 HIDE episodes, 0 REINIT flags, 377 localisation calls, 175/450 tracked. Pipeline time 129.95 s.

**Description**: The overlay is available on three of ten audit frames. The strongest-perspective frame and all five mask frames are hidden, so zero leakage is explained by absent overlays.

**Uncertainty**: All settings and seed 0 were transferred without personal-clip tuning. Dimensions and event ranges are approximate. Manual corners behind occluders are uncertain. Tracker jitter uses sparse GT motion interpolation. Rendered placement uses SIFT re-localisation and differs from saved-corner error. No full board exit/re-entry occurs, so re-entry recovery is not assessed.

Evidence and monitor: [source/overlay/failure](../../student_files/stage4_shared/stage4_ruijie_classical_evidence.png), [timeline](../../student_files/stage4_shared/stage4_ruijie_classical_timeline.png).
