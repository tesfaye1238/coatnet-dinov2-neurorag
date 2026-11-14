# Architecture

```
image [B,3,224,224] -> CoAtNet-0 (timm coatnet_0_rw_224, global-avg-pool) -> a [B,768] --\
                    -> DINOv2 ViT-S/14 (timm, img_size 224, CLS token)    -> b [B,384] --+-> concat [B,1152]
 -> SE gate: g = sigmoid(W2 relu(W1 z)), 1152->144->1152, out = z*g
 -> Linear(1152,512) ReLU Dropout(0.40) Linear(512,128) ReLU Dropout(0.30) Linear(128,4) -> logits
```

Logits are returned; softmax is applied only for probabilities and reporting (the training loss is cross-entropy on logits).

* The branch outputs are already pooled vectors, so the SE gate has no averaging over the feature dimension.
* The builders fail with `BackboneDimensionError` if a backbone does not give the reported pooled size. A projection added to reconcile
  dimensions would be a methodological change and must be documented.
* Parameter counts are computed from the instantiated model (`count_parameters`); none is hard-coded. Instantiated totals are about 0.085 M
  above the manuscript's (D16) while the differences between variants agree: SE r=8 adds 333,072 parameters, r=16 adds 167,112, and concat vs
  sum differs by 86,064 given a 384->768 projection.
* Ablations (`models/factory.py`): `coatnet_only`, `dinov2_only`, `concat_no_attention`, `concat_se_r16`, `sum_se_r8`, `proposed`.
  Extensions (not in the manuscript): `concat_mlp_control`, `PredictionEnsemble`.
* Baselines: GhostNetV2, ConvNeXt-T, Swin-T, MobileNetV2 (timm), ShuffleNetV2 (torchvision). ESPNetV2 raises `BaselineUnavailable` until a
  verified implementation with provenance and licence is registered; no other model is substituted under its name.
* Weights: hub ids `timm/coatnet_0_rw_224.sw_in1k` and `timm/vit_small_patch14_dinov2.lvd142m`. The commit shas at download are in
  `pretrained_provenance.json` (timm does not pin them).

## Attribution

Grad-CAM and Grad-CAM++ use the fused model's class logit and one branch layer: CoAtNet `stages[-1]` (feature map) or DINOv2
`blocks[-1].norm1` (tokens; class token dropped, 16x16 patch grid). The OUTPUT of the last transformer block is not used: with class-token
pooling it carries no patch-token gradient and would give a blank map (the extractor raises). A one-branch map is not an explanation of the
fused decision.
