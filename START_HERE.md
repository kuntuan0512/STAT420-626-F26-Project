# STAT 626 · 100-patient MRI segmentation

Upload the notebook in this folder to Google Colab. In the dataset folder its filename is **STAT626_100_Patients_2D_Colab.ipynb**. It contains the revised three-slice workflow and embeds the Python source and frozen patient selection.

1. Put the existing **BraTS100_2D.zip** in **My Drive/STAT626** or set DATA_SOURCE in section 2 to its Drive path. The earlier 200-patient archive and original shard manifest are also supported.
2. Select **QUICK_CHECK** and run sections 1–5 to see the EDA figures. Continue through section 9 to check training and evaluation on six examinations.
3. Select **STUDY_100_AMCA** and rerun sections 2–5 to audit all 100 examinations. Use a GPU runtime and complete sections 6–8 to train M0, M1 and M2. Enable held-out evaluation in section 9 after selecting checkpoints on validation.
4. Section 10 optionally trains and evaluates the channel-concatenation ablation with M1 supervision. It uses the same split and its own output folder.

The notebook runs the Python scripts for you. Outputs are saved by profile under **My Drive/STAT626_100patients_AMCA_v1**. After a runtime reset, rerun setup and the training cell to resume from the last completed epoch.

| Item | Setting |
|---|---|
| Groups | 100 filename-derived groups, one complete examination each |
| Split | 70 train, 15 validation, 15 test |
| MRI | Four sequences, each 240 × 240 × 155 in the audited cohort |
| Model input | 12 image planes (three per sequence) and four availability maps |
| Model output | Four tissue logits per pixel of the center slice |
| Fusion | Four-head mean-query attention with unavailable keys masked |
| M0 | Complete-view cross-entropy plus foreground soft Dice |
| M1 | Average the supervised complete and single-missing losses |
| M2 | M1 plus a fixed 0.1 consistency weight on detached complete-to-missing KL |
| Full training | Width 16, seed 11, batch 4, 16 target slices per patient per epoch, up to 10 epochs |

The existing folder and archive names reflect the original packaging. The current model is 2.5D with 2D convolutions. Earlier single-slice checkpoints use a different architecture and require their original notebook.

The three-slice figure is **three_slice_explained.png**. **LOCAL_SUBSET_AUDIT.json** records the full QC and contains local Windows paths. Colab builds its own audit for its staged data.
