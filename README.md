# SegmetationRobustness
Due to the increase in computational resources and accessibility of data, an increase in large, deep learning models trained on copious amounts of data using self-supervised or semi-supervised learning have emerged. *These foundation* models are often adapted to a variety of downstream tasks like classification, object detection, and segmentation with little-to-no training on the target dataset. In this work, we perform a **robustness analysis** of Visual Foundation Models (VFMs) for segmentation tasks and compare them to supervised models of smaller scale. We focus on robustness against **real-world distribution shift** perturbations. We benchmark **four** state-of-the-art segmentation architectures using **2** different datasets, **COCO** and **ADE20K**, with **17** different perturbations with 5 severity levels each. We find interesting insights that include (1) VFMs are not robust to compression-based corruptions, (2) while the selected VFMs do not significantly outperform or exhibit more robustness compared to non-VFM models, they remain competitively robust in zero-shot evaluations, particularly when non-VFM are under supervision and (3) selected VFMs demonstrate greater resilience to specific categories of objects, likely due to their open-vocabulary training paradigm, a feature that non-VFM models typically lack. We posit that the suggested robustness evaluation introduces new requirements for foundational models, thus sparking further research to enhance their performance. 

# Real-World Perturbed Segmentation Datasets
## Download Data
The data for COCO-P and ADE20K-P are available for download [here](https://www.crcv.ucf.edu/data1/segmentation_robustness_benchmark/).

## COCO
Directory Structure:
```bash
|-- <corruption>
|   |-- <severity>
|   |   `-- coco
|   |       |-- annotations
|   |       |   |-- captions_val2017.json
|   |       |   |-- instances_val2017.json
|   |       |   |-- panoptic_caption_val2017.json
|   |       |   |-- panoptic_val2017.json
|   |       |   `-- person_keypoints_val2017.json
|   |       |-- panoptic_semseg_val2017
|   |       |-- panoptic_val2017
|   |       `-- val2017
```

## ADE20K
Directory Structure:
```bash
|-- <corruption>
|   `-- <severity>
|       `-- ade
|           `-- ADEChallengeData2016
|               |-- ade20k_instance_val.json
|               |-- ade20k_panoptic_val
|               |-- ade20k_panoptic_val.json
|               |-- annotations
|               |   `-- validation
|               |-- annotations_detectron2
|               |   `-- validation
|               |-- annotations_instance
|               |   `-- validation
|               |-- images
|               |   `-- validation
|               |-- objectInfo150.txt
|               `-- sceneCategories.txt

```

## Licensing
COCO is under Creative Commons Attribution 4.0  and ADE20K is under BSD 3-Clause License. Please see [COCO](https://cocodataset.org/#home) and [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) for more details. Licensing information for each model are located in [models/](models).

# Models
See [README.md](models/README.md) for installation instructions and model details.

# Related Asset & Acknowledgment
Our work is relies on the open-source models built and inspired by several assets. 
We gratefully thank the authors for their open-source projects that 
allowed this benchmark to be possible:

* ODISE: https://github.com/NVlabs/ODISE/
* Mask2Former: https://github.com/facebookresearch/Mask2Former/
* MaskDINO: https://github.com/IDEA-Research/MaskDINO
* Detectron2: https://github.com/facebookresearch/detectron2/
* mmsegmentation: https://github.com/open-mmlab/mmsegmentation
* COCO panoptic: https://github.com/cocodataset/panopticapi

