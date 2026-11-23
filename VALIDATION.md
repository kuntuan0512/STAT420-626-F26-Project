# Verification of the three-slice notebook and scripts

- Full cohort audit completed locally on all 100 examinations. All 100 passed, with 400 MRI volumes and 100 reference masks. The patient-group split contains 70 training, 15 validation and 15 test groups.
- The attention test compared the four-head output with a direct evaluation of the equations. It also checked missing key weights, invariance to withheld pixels, matching initialization across regimes, gradient flow, and three-slice boundary handling.
- A CPU QUICK_CHECK trained M0, M1 and M2 for one epoch each on four patients. It validated on one patient and evaluated one held-out patient in the complete and four single-missing conditions. The saved CSV has one row for each of 3 models × 5 input conditions × 3 regions.
- Every notebook code cell parses as Python. The embedded archive passes integrity checks and contains byte-for-byte copies of the shipped sources and selection manifest.

The full 100-patient training, held-out comparison, and optional concatenation ablation remain pending. The quick-check test split contains one patient, which is insufficient for a between-patient confidence interval. CUDA execution of the revised notebook remains unverified.
