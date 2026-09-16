# Obstacle contour experiment, 2026-09-06

User-authorized experiment on YSH; no production runtime replacement.

Final primary data: SmokeBasement (Zenodo 15173055), actual thermal frames from
a smoke-drill basement. Run3 middle window: 32 distinct training PNGs spanning
24 seconds. Run5 middle window: 120 distinct validation PNGs resampled by nearest
timestamp at 10 fps over 12 seconds. Context photos and author description confirm
the smoke-drill setting; local smoke concentration is unmeasured. Same basement,
different runs, not independent-site generalization. IFSI remains auxiliary.

User clarification: the main condition is smoke-obscured visibility viewed through
a real thermal camera. The initial RGB experiment is auxiliary only. Main run now
uses IFSI Video 8 (2).mp4 for adaptation (exclude first 4 seconds of fade-in from
training sampling), and IFSI Video 4 (2).mp4 as a video-file holdout. Their capture
sessions may overlap: this is NOT a certified scene/session-disjoint test. Dense
smoke severity and synchronized RGB/thermal pairing are not established by these
files. No RGB-to-thermal simulation is used. Source OSD regions are ignored in
training loss and the same repository OSD suppression is applied in comparisons.

Goal: preserve obstacle silhouettes and passage geometry without panels, labels,
boxes, or a requirement to classify objects. Edge sparsity alone is not success.

Compare three approaches: (1) unchanged TEED with better rendering and contour
selection (preferred first); (2) conservative TEED adaptation using multiscale
pseudo targets, explicitly not ground-truth supervised obstacle training;
(3) independently pretrained PiDiNet as an available edge accuracy challenger.
Research newer LED-Net and thermal-specific depth, but do not infer superiority
from publication year or RGB benchmark scores.

1. Initial auxiliary run: Fire360 sample_3 adaptation, sample_4 held-out RGB video.
   Record hashes, clip intervals, frame counts and source URLs. Hold out repository
   thermal images entirely in the RGB experiment. The IFSI still is part of the
   thermal held-out source video and is not an extra independent test. Do not turn
   still images into purported field video.
2. Test contour hysteresis: retain weak continuations joined to strong contours,
   reject isolated weak texture, preserve short strong obstacles, handle empty maps.
3. Generate baseline, one-pixel TEED ablation, structural TEED and adapted TEED
   outputs with matching 320x240 resolution and frame timing. Keep output free
   of added text/UI. Use source at 55% brightness in candidates and separately
   retain edge masks so rendering does not masquerade as model improvement.
4. Fine-tune only TEED fusion head initially, deterministic seed, balanced soft
   BCE to teacher contours from reduced-scale input; freeze feature extractor.
   Record initial/final loss and changed weights. No obstacle-recall claims from
   pseudo-label agreement. Keep baseline and all checkpoints.
5. Run PiDiNet using official preprocessing/checkpoint. Measure warm CPU model
   and pipeline latency independently from video decoding/encoding. Measure
   edge occupancy and motion-aligned disagreement as diagnostics only.
6. Inspect representative frames, decode all output videos, record limitations,
   create a Korean report and open saved videos in a review window.

Required next-stage true fine-tuning: scene-separated thermal videos, human
annotations of obstacle silhouettes, floor-wall/door/step boundaries and ignore
regions for smoke/OSD. Balanced boundary BCE plus Dice and motion-masked temporal
consistency, with extra weight on missed nearby thin obstacles. Evaluate boundary
recall, passage geometry and human avoidance judgments, not just fewer edges.
