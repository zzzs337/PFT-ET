# Data preparation and evaluation

Dataset sources and pretrained resources are listed in [Resources](resources.md). Prepare input manifests and annotations using [Data preparation](../data/README.md); field definitions are in [Data dictionary](DATA_DICTIONARY.md).

Store actual split membership in JSON before generating prefixes. Training copies the split and manifest into its run directory. Subject IDs define UP-Fall uncertainty clusters, while physical IDs link synchronized camera streams.

Use `scripts/export_evaluation_records.py` to export per-run, stream, event, alarm, and method/seed/subject/camera monitoring tables. Use `scripts/recompute_evaluation.py` to check metrics and exposure against saved predictions.

Use `scripts/summarize_dataset.py` to export actual split membership and physical-event/camera mapping. Use `scripts/validate_research_inputs.py` before a research run to check that the configured manifest, split, arrays, backbone checkpoint, and event annotations are present and internally consistent. `examples/` contains synthetic records with the same fields as these exports.

Execution commands are in [Reproducibility](REPRODUCIBILITY.md), and generated fields and units are in [Output schemas](OUTPUTS.md).
