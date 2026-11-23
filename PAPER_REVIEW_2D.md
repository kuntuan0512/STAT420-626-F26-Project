# Literature and the three-slice attention design

The main comparison tests whether consistency adds value to supervised modality dropout. M0, M1 and M2 share the same availability-masked cross-attention and U-Net. The separate M1 ablation tests this fusion layer against concatenated, availability-gated stem features.

| Paper | Connection to the project |
|---|---|
| [CKD-TransBTS](https://arxiv.org/abs/2207.07370) | Its modality-correlated attention motivates learning how to combine correlated sequences. We pool the available features into a query at each pixel and retain four separate modality stems. |
| [CMAF-Net](https://qims.amegroups.org/article/view/125898/html) | Its focus on incomplete MRI motivates sequence dropout and evaluation by missing modality. Our four held-out conditions each remove one sequence in turn. |
| [Clinical Knowledge-Based Hybrid Swin Transformer](https://www.techscience.com/cmc/v76n3/54381/html) | Its paired sequence streams motivate modality-specific encoding. Our shared query combines all available streams. |
| [MSegNet](https://link.springer.com/article/10.1007/s44196-025-00787-7) | Its multi-view processing provides context for our choice to use three adjacent axial planes, with a prediction for the center plane. |
| [CMHF-3DNet](https://pubmed.ncbi.nlm.nih.gov/42225491/) | The authors' abstract describes voxel-wise cross-modal transformer fusion and hierarchy-aware learning. We use local cross-modal fusion and define nested regions by unions of four exclusive tissue labels. |

Each modality stem maps three planes to a feature map. At each pixel, the mean feature of the observed modalities supplies the query. Each modality supplies a key and a value. A learned modality embedding is added to its key. The softmax masks unavailable keys across four heads. The projected attention output is added to the available mean, and the four availability maps enter the U-Net alongside this residual.

The complete-input prediction is used as a detached target for M2's consistency penalty. Both the complete and single-missing predictions retain segmentation supervision. The penalty weight is fixed at 0.1 for this study.

The cited architectures use different datasets, splits, missing-input conditions and spatial models. Their results motivate the experiments. The 100-patient comparison uses paired measurements from our own split.
