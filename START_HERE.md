# Simplified study: 100 patient groups, 2D cross-attention

Open `STAT626_100_Patients_2D_Colab.ipynb` in [Google Colab](https://colab.research.google.com/) using File > Upload notebook. Choose a GPU runtime. Run the cells with QUICK_CHECK first, then change PROFILE to STUDY_100_2D and rerun cells 2–8. Enable held-out evaluation in cell 9 after freezing your choices.

The notebook is self-contained: it embeds the source code and `subset_selection.json`. The readable Python files are provided separately. If you edit those files, load them explicitly instead of running the embedded-source cell, which restores the packaged snapshot.

## What changed

| Item | Simplified study |
|---|---|
| Cohort | 100 filename-derived patient groups; one complete examination per group |
| Selection | Seed 626; SHA-256 ranking of patient IDs before reading segmentation labels |
| Repeated examinations | First lexicographically sorted complete examination per group |
| Split | 70 training, 15 validation, 15 test groups after QC |
| Input | One aligned axial slice per modality: 4 MRI channels + 4 availability maps |
| Image dimensions | Native axial dimensions; 240 × 240 in the audited dataset examples |
| Models | A0 masked mean fusion; A1 cross-modal attention |
| Network | 2D U-Net, base width 16, four output classes |
| Training | CE + Dice on complete and randomly single-missing inputs; no consistency loss for A0/A1 |
| Budget | Up to 10 epochs, seed 11, batch 4, 16 sampled training slices per patient per epoch |
| Validation | Every 2 epochs; all slices and all five input conditions |
| Evaluation | Stack independently predicted slices for patient-volume WT/TC/ET Dice and HD95 |

The previous model was 2.5D (2D convolutions with three neighboring slices). This version uses one slice per modality and 8 input channels. The original NIfTI files stay in 3D storage; no conversion to lossy JPEG/PNG is required. Normalization remains per MRI volume. The model has no neighboring-slice context.

## Dataset and counts

The saved source inventory contains **1,251 examinations and 1,133 filename-derived groups**, not 1,600 verified patients. Patient grouping still needs independent identity verification. The fixed selection has 100 examinations, **400 MRI volumes and 100 segmentation masks**. No naturally missing MRI file is selected. Simulated missingness gives 100 cases per withheld modality, or 15 per condition in the held-out test partition. Repeated conditions and slices are not independent patients.

`subset_selection.json` freezes all selected case IDs and file sizes. `select_subset.py` can reproduce the selection. Full_audit in the subset EDA refers to all selected examinations, not the original full cohort. If QC excludes a selected case, the full-study workflow stops rather than silently replacing it.

## Two ways to supply data to Colab

1. Use the existing 52 dataset ZIP shards and `BraTS_Full_Manifest.json` under `My Drive/STAT626`. The notebook extracts only the fixed selection. It may still transfer most original shards because the chosen groups are distributed throughout them.
2. Upload **BraTS100_2D.zip** to `My Drive/STAT626`. The notebook prefers that smaller archive when it exists. The separately supplied `.manifest.json` contains its SHA-256 checksum. The notebook checks ZIP CRC on extraction and matches selected filenames/file sizes; original shard mode also verifies supplied shard SHA-256 hashes.

Default DATA_SOURCE is `/content/drive/MyDrive/STAT626`. You can change it to an explicit ZIP path or extracted dataset directory. Hosted Colab cannot access Windows D: paths. All copying is local/within your own Drive; no dataset is published or uploaded automatically.

If your earlier `BraTS200_2D.zip` is already on Drive, you can reuse it by setting DATA_SOURCE to `/content/drive/MyDrive/STAT626/BraTS200_2D.zip`. The new notebook extracts only its frozen 100-group selection from that archive. Uploading the new smaller ZIP is optional in that case.

The new local folder is `Dataset/STAT626_2D_100/`. Original code, checkpoints and raw scan folders remain available. Do not resume original 2.5D checkpoints with these 2D models.

## Cross-attention and the sketch

Each modality has a one-channel convolutional stem. At each coarse spatial location, features from modality m form queries, and features from the other available modalities form keys and values:

`Attention(Q_m, K_other, V_other) = softmax(Q_m K_other^T / sqrt(d)) V_other`

Missing streams are masked. The attention updates are fused into a feature map and passed to the 2D U-Net. A0 uses the same stems/backbone with masked mean fusion; A1 adds the attention block. This is local attention across modalities, not full spatial attention between every pixel. It follows the Q/K/V concept in the whiteboard image; it is a small adaptation rather than a replication of a published 3D system.

Background, NCR, edema and enhancing tumor are four exclusive labels. Evaluation derives WT={1,2,3}, TC={1,3}, ET={3}. These overlapping regions remain nested by construction.

## Interpretation and recovery

The main comparison is A1 minus A0 on the same held-out groups under four single-missing conditions. This is an exploratory course project. Fifteen test groups and one training seed provide limited precision; patient bootstrap intervals do not capture training-seed variability. Do not claim attention improves accuracy until matched held-out evaluation supports it. Select settings using training/validation only.

Outputs and epoch checkpoints persist under `My Drive/STAT626_100patients_2D_v1/<profile>/`. After a reset, rerun cells 1–7 with the same settings to resume from the last completed epoch. Interrupted evaluation starts a new attempt; completed evaluation is reused. Colab Pro resource availability varies; no training completion time is guaranteed.

## Source files and standalone use

- `eda.py`: QC, data counts, patient splits, figures and single-slice input construction.
- `design_network.py`: true 2D mean-fusion and cross-attention networks.
- `train_model.py`: matched training, checkpoint selection and held-out evaluation.
- `select_subset.py`: deterministic cohort selection and subset staging/ZIP creation.
- `colab_workflow.py`: notebook setup, profile settings and recovery.

```bash
python eda.py --data-root /path/to/full/cohort --subset-manifest subset_selection.json --output /path/to/new/eda
python design_network.py --self-test
python train_model.py train --eda-dir /path/to/new/eda --output /path/to/new/models --device cuda
python train_model.py evaluate --eda-dir /path/to/new/eda --run-dir /path/to/new/models --output /path/to/new/test --device cuda
```

Research motivation: [CMAF-Net](https://qims.amegroups.org/article/view/125898/html). Implementation API: [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html). Runtime guidance: [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Five-paper review and optional consistency experiment

Read [PAPER_REVIEW_2D.md](PAPER_REVIEW_2D.md) for all five suggested papers, the mapping to this 2D model, attention and consistency equations, limitations and presentation wording. The same review is embedded in the notebook before the GPU preflight. CMHF-3DNet is reviewed at abstract level because its publisher full text was inaccessible.

The default remains the smaller A0/A1 comparison. Notebook section 11 optionally runs A0/A1/A2 with the same 100 groups, split, width, seed and 10-epoch budget. A2 adds `0.1 * KL(stopgrad(full) || missing)` to the shared supervised loss. Lambda 0.1 is a fixed preset, not a tuned optimum. This optional experiment uses `regularization_ablation_lambda0p1` under STUDY_BASE and leaves existing default-study checkpoints intact.

Set RUN_REGULARIZATION=True to train that comparison after completing the full-study setup/EDA. Set RUN_REGULARIZATION_TEST=True separately after freezing the protocol. Both switches default to false. Rerunning the optional training cell resumes its own saved epoch checkpoints. Adding an experiment after reviewing held-out results must be described as exploratory.
