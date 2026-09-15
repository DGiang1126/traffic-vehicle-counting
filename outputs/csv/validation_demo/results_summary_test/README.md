# Results Summary Validation

`build_results_summary.py` was tested without using official experiment data.

Passing cases:

- `valid_manifest.csv`: 6 configurations, including E1/E2/E3 and an extra E5-style experiment.
- Project-root and manifest-relative summary paths both resolve.
- Blank FPS and blank `time_mae` are accepted.
- Extra metadata columns (`video_set`, `notes`) are preserved.
- `real_outputs_manifest.csv`: combines 4 summary files produced by the real `evaluate_counting.py` validation runs.
- `no_fps_manifest.csv`: FPS column is optional.

Expected-error cases:

- duplicate experiment ID -> rejected.
- non-numeric FPS -> rejected.
- missing summary file -> rejected.
- manifest/summary experiment mismatch -> rejected.
- summary missing required metrics -> rejected.
