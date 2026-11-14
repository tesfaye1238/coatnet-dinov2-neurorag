# Implementation decisions

Status codes: **R** reported in the manuscript; **V** verified by supplied code or a public record; **N** new implementation decision of this
repository; **P** proposed extension (not part of the original work); **U** unresolved. No original experiment code or record was supplied,
so nothing here is "V" with respect to the authors' own code. Everything newly written is a **manuscript-based implementation**.

## Data
| Item | Status | Detail |
|---|---|---|
| Four classes glioma / meningioma / no tumor / pituitary, alphabetical index order | R | `constants.CLASS_NAMES`; mapping stored in every checkpoint and verified on load. |
| Sources figshare, SARTAJ, Br35H | R | Registry records what was verified. figshare verified (md5s, licence, patient ids); SARTAJ and Br35H URLs only from the Zenodo description (login required, licence unverified). |
| figshare `.mat` -> PNG by per-image min-max to uint8 | V (source README) | The README gives this conversion; it discards the original dynamic range. |
| Patient ids | V for figshare (`cjdata.PID`, source-documented); missing elsewhere | Missing ids stay missing; heuristic filename groups are labelled `heuristic_filename` and never verified. |
| Group-stratified splitting, group integrity above target counts | N | Realised counts are reported against Table 1 as differences. |
| Validation drawn from each fold's training pool (10% of the pool) | N | 562 / 5,617 in the manuscript is about 10%. |

## Preprocessing and augmentation
| Item | Status | Detail |
|---|---|---|
| RGB, bilinear resize to 224, scale to [0,1], ImageNet normalisation | R | `data/transforms.py`. |
| Horizontal and vertical flip 0.5, rotation ±15, zoom 0.90-1.10, brightness ±10%, noise sigma 0.01, erasing p 0.10 | R | Defaults in `AugmentationConfig`. |
| Order, rotation/zoom probability 1, bilinear interpolation, fill 0, erasing area 0.02-0.33 and ratio 0.3-3.3, noise in [0,1] domain with clipping, multiplicative brightness | N | `UNSPECIFIED_AUGMENTATION_DETAILS` lists them; not presented as recovered. |
| Vertical-flip-off sensitivity | P | `configs/sensitivity_no_vflip.yaml`, separate experiment. |

## Architecture
| Item | Status | Detail |
|---|---|---|
| CoAtNet-0 (768-d pooled), DINOv2 ViT-S/14 (384-d CLS), concat 1,152, SE r=8 (1152->144->1152), head 1152-512-128-4 with dropout 0.40 / 0.30 | R | `models/`. |
| timm `coatnet_0_rw_224.sw_in1k` as "CoAtNet-0" | N (U) | Dimension 768 verified with real weights; identity with the authors' implementation unverified. Parameter totals are 0.085 M above the reported ones (see D16). |
| timm `vit_small_patch14_dinov2.lvd142m`, `img_size=224` (position embeddings interpolated) | N (U) | Dimension 384 verified. |
| Hub commits at download | V | CoAtNet `6686c542...`, DINOv2 `4610ca14...` in `docs/pretrained_provenance.json`; timm does not pin them. |
| Trainable policy | N | Both branches fully fine-tuned by default; `frozen` and `last_n` are options. |
| SE gate returns `z * sigmoid(W2 relu(W1 z))` on the pooled vector | R | No averaging across the feature dimension. |
| Sum-fusion ablation: DINOv2 384 -> 768 linear projection, then add | N, supported | Reported parameter gap (0.09 M) is reproduced (0.086 M); projecting 768 -> 384 would give 0.39 M. Consistent, not proof. |
| Capacity-comparable MLP control (same two linears, residual, no gate) | P | `concat_mlp_control`. |
| Prediction ensemble (mean of softmax) | P | `PredictionEnsemble`. |
| Baselines | R (names) / N (variants) | timm GhostNetV2 1.0, ConvNeXt-T, Swin-T, MobileNetV2 1.0; torchvision ShuffleNetV2 x1.0. ESPNetV2 is unavailable and raises. |

## Training
| Item | Status | Detail |
|---|---|---|
| AdamW, batch 32, peak LR 3e-4, weight decay 5e-4, 5-epoch warmup, cosine to 1e-6, max 100 epochs, patience 10 on validation loss, weighted CE `N/(K n_c)` | R | `training/`. |
| Per-iteration schedule, linear warmup from peak/warmup_steps, cosine horizon = max_epochs | N | Not affected by early stopping. |
| Early stopping on the class-weighted validation loss | N | The weighted loss is the stated loss. |
| Mixed precision off by default, no gradient clipping | N | Options. |

## Evaluation
| Item | Status | Detail |
|---|---|---|
| Accuracy, precision, recall, F1 (macro), confusion matrix, OvR ROC-AUC, AP | R | From saved predictions only. |
| OvR specificity and MCC, multiclass MCC reported separately | R/N | Distinguished in the code and reports. |
| Image-level bootstrap intervals, paired bootstrap for differences, exact McNemar, Holm over a declared family | N | The original CI and test procedures are unknown (D08-D10). |
| ECE 15 equal-width bins, Brier, NLL, AURC, coverage at 1% risk | N | Definitions in `evaluation/calibration.py`. |
| Perturbation severities | R (Table S2 values) / N (functional forms) | Rician construction, bias-field model, motion kernel are new. |
| FLOPs by `FlopCounterMode`, latency batch 1 after warm-up, no energy | N | Convention printed with the result. |

## Explainability
| Item | Status | Detail |
|---|---|---|
| Grad-CAM and Grad-CAM++ of the fused model's logit through one branch layer | N | CoAtNet `stages[-1]`; DINOv2 `blocks[-1].norm1`. The last block's output carries zero patch-token gradient under class-token pooling (tested), so it is rejected. |

## RAG
| Item | Status | Detail |
|---|---|---|
| Qdrant, Qwen2.5, category-conditioned retrieval, structured background summary | R | Names only. |
| BGE-base-en-v1.5, 768-d, cosine, 512/64 chunks, top-10 -> top-5, category filter, no reranker, Qwen2.5-7B-Instruct greedy 512 tokens | P (U) | Labelled "(proposed)" in Supplementary S6; candidate settings only. |
| Passage-id existence check, schema validation, injection screening, insufficient-evidence handling, optional uncertainty statement, withhold-on-invalid | N | `SAFEGUARD_PROVENANCE` marks each; only grounding and constrained prompting are R. |
| Demonstration collection | N | 8 synthetic test passages, labelled, not a reconstruction of the 1,842-entry collection. |
