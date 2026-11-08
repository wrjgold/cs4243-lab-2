# Overlay Scorecard — stage4_sean

**Clip**: stage4_sean, 15 s, 1280 x 720 at 30 fps. Landscape textured blue and pink poster with outer black frame on a wall. First-frame annotated area 16.80%. Event ranges: {"pan_and_distance_approx":[0,225],"board_occlusion_approx":[233,313],"viewing_angle_change_approx":[330,449],"board_clipping_near_end_approx":[378,449]}.

**Initialisation**: Descriptor-valid keypoints reference/frame 0: 988/861. Matches 86, inliers 40 (0.465), median residual 0.790 px.

**Geometry**: independent: median saved-corner error 0.674% on 4/10 finite frames, success 40% at 2%, unavailable 6/10. flow_assisted: median saved-corner error 0.670% on 4/10 finite frames, success 40% at 2%, unavailable 6/10.

**Temporal**: independent: tracker jitter 2.623 px, drift at least 16 frames after reinitialisation n/a on 0 finite audit frames, longest localisation interval 1 frames. flow_assisted: tracker jitter 2.455 px, drift at least 16 frames after reinitialisation n/a on 0 finite audit frames, longest localisation interval 10 frames.

**Occlusion**: independent: IoU 0.248, F1 0.397 over 1/3 nonempty masks, hidden 2, missing 0. Finished-video leakage 20.80% over all 3 nonempty masks. Coverage 94.753% on 2 overlay-present mask frames. Direct alpha leakage with/without masking 19.32%/33.33%, same saved geometry and HIDE policy. flow_assisted: IoU 0.248, F1 0.397 over 1/3 nonempty masks, hidden 2, missing 0. Finished-video leakage 20.80% over all 3 nonempty masks. Coverage 94.701% on 2 overlay-present mask frames. Direct alpha leakage with/without masking 19.32%/33.33%, same saved geometry and HIDE policy.

**Monitor**: independent: 335/450 HIDE frames, 26 HIDE episodes, 0 REINIT flags, 450 localisation calls, 115/450 tracked. Pipeline time 659.64 s. flow_assisted: 330/450 HIDE frames, 25 HIDE episodes, 0 REINIT flags, 409 localisation calls, 120/450 tracked. Pipeline time 542.82 s.

**Description**: The overlay is available on only four of ten audit frames. Frame 300 paints much of the purple foreground object, while frame 270 is hidden after localisation failure.

**Uncertainty**: All settings and seed 0 were transferred without personal-clip tuning. Dimensions and event ranges are approximate. Manual corners behind occluders are uncertain. Tracker jitter uses sparse GT motion interpolation. Rendered placement uses SIFT re-localisation and differs from saved-corner error. No full board exit/re-entry occurs, so re-entry recovery is not assessed.

Evidence and monitor: [source/overlay/failure](../../student_files/stage4/stage4_sean_classical_evidence.png), [timeline](../../student_files/stage4/stage4_sean_classical_timeline.png).
