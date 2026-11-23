# Verification: 100-patient-group 2D study

Verified November 17, 2026. The scientific workflow is the same as the earlier 200-group version, with a newly frozen 100-group cohort and separate output paths.

## Real dataset checks

- Exactly 100 distinct filename-derived groups, one complete examination each, selected reproducibly with seed 626 before inspecting segmentation labels.
- All selected examinations belong to the earlier 200-group selection.
- Full QC: 100 passed; zero failures/exclusions.
- 400 MRI volumes and 100 segmentation masks; zero naturally absent modality files.
- Shapes: {"240x240x155": 100}.
- Patient split: 70 training, 15 validation, 15 test; no group overlap.
- Subset ZIP: 500 NIfTI files, 960,793,469 bytes; all ZIP CRC checks passed.
- ZIP SHA-256: `6e3ad3b729b10c98e3db02f7b898db90ea3bb2fd636f79bded5706a2e960892c`.

`LOCAL_SUBSET_AUDIT.json` records local Windows-path evidence. Colab reruns QC with its own runtime paths; do not train in Colab using the local Windows-path manifest.

## Code and notebook checks

All notebook code cells compile. Embedded source files match the delivered Python files and selection manifest. Both the main A0/A1 command and optional A0/A1/A2 fixed-lambda command parse with the intended settings. The optional experiment remains disabled by default.

AST comparison confirms that the EDA, network and training computations are unchanged from the verified 200-group version; only descriptive cohort wording differs in those scripts. The model still uses single-slice 2D inputs, width 16, 10 epochs, seed 11, and 16 sampled slices per training patient per epoch. The 2,000 bootstrap-resample setting is retained. Existing CPU model tests apply; full training was not repeated just to change cohort size.

The new automatic BraTS100_2D.zip filename and staged-data reuse were checked on a tiny synthetic fixture. Actual 100-group selection, archive membership/integrity, full QC and frozen patient split were verified separately.

No full 100-group training or new CUDA execution has been performed here. The notebook runs a GPU preflight in Colab. Patient identity remains inferred from filenames, and 15 test groups with one trained seed support an exploratory comparison. Patient bootstrap intervals do not capture training-seed variability. The optional nbformat validator is not installed locally; notebook structure and executable cells were checked directly.
