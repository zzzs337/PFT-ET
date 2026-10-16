# Data dictionary

## Input identifiers

Each camera stream has a unique `sequence_id`. `physical_id` links views of the same activity/trial, and `subject_id` groups recordings from one subject. `camera` identifies the view; `activity` and the Boolean `fall` describe the recording.

## Annotations and arrays

`onset_seconds`, `impact_seconds`, and `completion_seconds` are relative to the aligned stream start. Fall completion supplies event-time labels; first impact supplies the pre-impact reference. ADL event times are null. `excluded_intervals` contains half-open time intervals.

The raw manifest references source images/video, original timestamps, and pose arrays. The processed manifest references RGB uint8 arrays and normalized 17-joint skeleton arrays at 20 FPS. Complete formats and preparation commands are in [Data preparation](../data/README.md).

## Data partitions

A split JSON contains actual `train`, `val`, and `test` sequence-ID lists plus available seed and provenance fields. All UP-Fall streams of one subject remain in one partition. UR Fall partitions complete Camera 0 sequences.

## Generated records

[Output schemas](OUTPUTS.md) defines prediction, event, alarm, exposure, calibration, reference-feature, case, and timing records, including their units and null-value conventions.
