# Data contract and preparation

Obtain full UP-Fall RGB data (17 subjects; both views) and UR Fall Camera 0 from the sources in `docs/resources.md`. Keep original timestamps and the state annotations needed to locate completion. The competition-only UP-Fall subset is not the paper dataset. First impact must be additionally annotated, with provenance; the code cannot derive a physical contact timestamp from the completion label.

## Raw manifest

Raw JSONL contains one camera stream per line. Paths are relative to the raw manifest's directory. Seconds are relative to the first frame; for synchronized views, align clocks to a common physical event reference before preparing them. `physical_id` identifies the same activity/trial across cameras, while `sequence_id` is unique per camera. `subject_id` groups every UP-Fall activity/trial/view.

```json
{"sequence_id":"subject01_activity01_trial01_cam1","physical_id":"subject01_activity01_trial01","subject_id":"subject01","camera":"1","activity":"fall_forward","fall":true,"frame_list":"subject01_activity01_trial01_cam1/frames.txt","timestamps":"subject01_activity01_trial01_cam1/timestamps.csv","skeleton":"subject01_activity01_trial01_cam1/pose.npy","bboxes":"subject01_activity01_trial01_cam1/bboxes.npy","onset_seconds":3.0,"impact_seconds":3.6,"completion_seconds":4.0,"excluded_intervals":[]}
```

This is a schema example, not a dataset annotation. `frame_list` lists one image path per source frame. Alternatively use `video` with a video path. `timestamps` is a headerless one-column CSV of original seconds, strictly increasing and matching decoded frames; UR synchronization milliseconds must be converted to seconds first. Images must have fixed dimensions per stream. `skeleton` is source-frame `[N,17,3]` float32 **pixel x, pixel y, confidence**, produced by `extract_pose.py` or supplied from the matching pose extractor. Optional bboxes are `[N,4]` xyxy for one tracked person per frame; otherwise pose estimation uses the full image.

For ADL set `fall:false` and event times to null. Retain complete post-fall recording frames for transition targets that cross completion. State-to-boundary mapping follows Sec. 4.1: first laying after falling for UP-Fall; first lying-on-ground after falling/transient for UR Fall. Convert the source state annotations and independently reviewed impact boundaries to this manifest before preparation. Available annotation source and version fields are preserved.

Convert a structured annotation table to JSONL with:

```bash
python scripts/build_manifest.py --metadata annotations.csv --output data/raw/upfall/manifest.jsonl
```

The accepted columns and synthetic rows are shown in `examples/data/metadata.example.csv`. The converter validates identities, event ordering, frame-grid alignment, synchronized camera boundaries, and referenced source paths when `--check-paths` is used.

## Commands

```bash
python scripts/extract_pose.py --manifest data/raw/upfall/manifest.jsonl --pose-config /path/to/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py --pose-checkpoint /path/to/coco_checkpoint.pth --device cuda
python scripts/prepare_data.py --manifest data/raw/upfall/manifest.jsonl --output data/processed/upfall
python scripts/prepare_data.py --manifest data/raw/urfall/manifest.jsonl --output data/processed/urfall
```

Output per stream: resized RGB `rgb.npy` uint8 `[N,224,224,3]`, normalized `skeleton.npy` float32 `[N,17,3]`, 20-FPS `timestamps.npy`, and a processed JSONL manifest with mapped boundaries. Arrays use memory mapping. The preprocessor currently decodes a whole source stream in RAM; process independently split recordings rather than one concatenated long video. Files fail validation on missing timestamps, inconsistent frames, wrong joint count, event bounds or misaligned synchronized references.

## Split IDs

Provide the original split JSON when available:

```json
{"train":["stream_a"],"val":["stream_b"],"test":["stream_c"],"provenance":"original_partition"}
```

Every manifest stream appears exactly once. All UP-Fall views and trials of a subject share one partition; UR Fall uses one Camera 0 stream per complete physical sequence. Counts are enforced: 11/3/3 subjects or 49/10/11 sequences, stratified 21/28,4/6,5/6.

For a **new protocol-consistent partition**, use:

```bash
python scripts/make_splits.py --manifest data/processed/upfall/manifest.jsonl --dataset upfall --seed 3407 --output data/splits/upfall_generated3407.json
python scripts/make_splits.py --manifest data/processed/urfall/manifest.jsonl --dataset urfall --seed 3407 --output data/splits/urfall_generated3407.json
```

Point `data.splits` to these paths in a separate config. Generated IDs carry `provenance: generated_partition`; they are not the manuscript's original membership. Additional split sensitivity should preserve protocol and sizes, train three methods independently on each split, and choose thresholds on each corresponding validation partition.

## Monitoring accounting

Event evaluation uses Eq. (4) valid positions: complete observation, pre-completion for falls, and a fully observable H-second future horizon for ADLs. Exclusions invalidate a prefix if they intersect its observation window. Every emitted ADL position contributes a single 1/20-second cell; overlapping observation windows are not added as exposure. Warmup, end-of-sequence and annotated exclusions are included in the saved excluded-duration total. Predictions are reset over gaps; simultaneous UP-Fall views contribute separate camera-stream exposure. No new independent monitoring hours are created by rerunning the same stream with another seed.

## Dataset audit exports

`scripts/summarize_dataset.py` writes `dataset_statistics.csv`, `split_membership.csv`, and `event_camera_map.csv` directly from a manifest and split. These tables expose the counts and identifiers needed to check partition leakage and physical-event/camera mapping. Add `--check-protocol-counts` for the complete UP-Fall or UR Fall data used by the paper.
