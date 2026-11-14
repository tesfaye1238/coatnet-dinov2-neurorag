# Limitations

* **No result here is a measurement on the manuscript's data.** There is no training run, no real-data evaluation and no reproduction of the
  99.31% figure. A completed run would be a new experiment.
* The authors' code, images, split files, fold manifests, predictions, RAG collection, prompts and annotations were not supplied, and no
  reviewer correspondence was supplied beyond the issues listed in the task brief.
* The Zenodo record cited as the dataset is a model record (D01). The 7,022-image benchmark cannot be rebuilt exactly from public sources with
  the information available, and SARTAJ/Br35H need Kaggle terms and an unverified licence check.
* Patient-level independence can be established only for figshare (source patient ids). For other images the grouping is heuristic.
* The CoAtNet-0 implementation is timm's and may differ from the authors'; instantiated parameter totals are about 0.085 M higher.
* Unreported augmentation and optimisation details are reconstruction choices.
* PyTorch in the development environment is CPU-only. GPU, mixed precision, Docker and Linux CI were not exercised.
* ESPNetV2 has no verified implementation; no ESPNetV2 baseline exists here.
* Robustness perturbations are image-space approximations. Grad-CAM maps are branch-specific and are not segmentation masks.
* The retrieval and generation settings are candidate settings. The demo collection is synthetic. The extractive generator contains no model
  knowledge. Citation existence checks do not establish citation support or medical correctness, and no clinician ratings exist.
* Nothing here is clinically validated. The demo app labels all output experimental.
