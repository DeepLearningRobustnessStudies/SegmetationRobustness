Here we provide information and instruction on how to install and run the evaluated models for our benchmark study. 
All evaluation scripts are modified from the original authors to work with this dataset and for collecting/analyzing results.

# GroundedSAM
Based off of Segment-Anything and GroundedDINO:

**Segment Anything Model (SAM)** trained on a [dataset](https://segment-anything.com/dataset/index.html) of 11 million images and 1.1 billion masks.
**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

**GroundingDINO**
Official PyTorch implementation of ["Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"](https://arxiv.org/abs/2303.05499): the SoTA open-set object detector.
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/zero-shot-object-detection-on-mscoco)](https://paperswithcode.com/sota/zero-shot-object-detection-on-mscoco?p=grounding-dino-marrying-dino-with-grounded) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/zero-shot-object-detection-on-odinw)](https://paperswithcode.com/sota/zero-shot-object-detection-on-odinw?p=grounding-dino-marrying-dino-with-grounded) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=grounding-dino-marrying-dino-with-grounded) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=grounding-dino-marrying-dino-with-grounded)



## Installation
Original installation instructions is in [models/GroundedSAM/README.md](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/README.md)

```bash
module load anaconda3
module load cuda/11.7

conda create -n groundedsam python=3.9
source activate groundedsam
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia

git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
mv Grounded-Segment-Anything/ GroundedSAM/
cd GroudedSAM
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install --upgrade diffusers[torch]
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel


cd weights
# download the pretrained groundingdino-swin-tiny model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Regster datasets for eval
python models/Mask2Former/mask2former/data/datasets/register_ROBUSTNESS_coco_panoptic_annos_semseg.py
python models/Mask2Former/mask2former/data/datasets/register_ROBUSTNESS_ade20k_panoptic.py
```

## Running
Model types available are:
* swinT
* swinB

To run call, an example:
```bash
python eval_grounded_sam.py --model_type <model_type> --corruption <corruption> --severity <severity> --dataset <dataset>
```

# ODISE
**ODISE**: **O**pen-vocabulary **DI**ffusion-based panoptic **SE**gmentation exploits pre-trained text-image diffusion 
and discriminative models to perform open-vocabulary panoptic segmentation.

[**Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models**](https://arxiv.org/abs/2303.04803)
[*Jiarui Xu*](https://jerryxu.net),
[*Sifei Liu**](https://research.nvidia.com/person/sifei-liu),
[*Arash Vahdat**](http://latentspace.cc/),
[*Wonmin Byeon*](https://wonmin-byeon.github.io/),
[*Xiaolong Wang*](https://xiaolonw.github.io/),
[*Shalini De Mello*](https://research.nvidia.com/person/shalini-de-mello)
CVPR 2023 Highlight. (*equal contribution)

The model architecture relies on foundational models:
* [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
* [CLIP](https://github.com/mlfoundations/open_clip)
* [Mask2Former](https://github.com/facebookresearch/Mask2Former/)
* [Glide](https://github.com/openai/glide-text2im)

## Installation
Original installation instructions is in [models/ODISE/README.md](https://github.com/NVlabs/ODISE/blob/main/README.md).
Download the weights according to the [original repository](models/ODISE/README.md).

The following need to be added before builds. 
*NOTE: Do not run on PASCAL GPUs

Files added:
`models/ODISE/datasets/ROBUST_prepare_coco_semantic_annos_from_panoptic_annots.py`
`models/ODISE/third_party/Mask2Former/build/lib.linux-x86_64-cpython-39/mask2former/data/datasets/register_ROBUSTNESS_coco_panoptic_annos_semseg.py`


added `register_ROBUSTNESS_coco_panoptic_annos_semseg`  to `models/ODISE/third_party/Mask2Former/build/lib.linux-x86_64-cpython-39/mask2former/data/datasets/__init__.py`


```bash
module load anaconda3
module load cuda/11.7

conda create -n odise python=3.9
source activate odise
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia


# conda install -c "nvidia/label/cuda-11.6.1" libcusolver-dev
git clone https://github.com/NVlabs/ODISE.git
cd models/ODISE
cd third_party/Mask2Former
python setup.py build install
cd ../..
pip install -e .
```

For when making visualizations, I commented out several `register_<datasets>` 
from `datasets/__init__.py` and `data/__init__.py` because they are already registered.
This may cause problems later, but not sure. Mask2Former is where the datasets get registered
under `third_party/Mask2Former`.

## Running
Model types available are:
* caption
* label

To run call, an example:
```bash
python eval_odise.py --model_type <model_type> --corruption <corruption> --severity <severity> --dataset <dataset>
```

Results will be stored in `output/COCO_ODISE_<model_type>/<corruption>_<severity>/default`.

Evaluator is in passed in `configs/common/data/coco_panoptic_semseg.py`

To make any adjustments to the datasets, must re-install `mask2former`:
```bash
cd models/ODISE/third_party/Mask2Former
rm -rf build
rm -rf mask2former.egg-info/
pip uninstall mask2former
python setup.py build install
```

# Mask2Former
**Mask2Former**: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR 2022)

[Bowen Cheng](https://bowenc0221.github.io/), [Ishan Misra](https://imisra.github.io/), 
[Alexander G. Schwing](https://alexander-schwing.de/), 
[Alexander Kirillov](https://alexander-kirillov.github.io/), [Rohit Girdhar](https://rohitgirdhar.github.io/)

[[`arXiv`](https://arxiv.org/abs/2112.01527)] [[`Project`](https://bowenc0221.github.io/mask2former)] [[`BibTeX`](#CitingMask2Former)]
#### Installation and Set Up

Model weights can be found in [models/Mask2Former/MODEL_ZOO.md](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md)

Changes made before installation: 

## Instructions
Original installation instructions is in [models/Mask2Former/INSTALL.md](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md).

detectron2 has some issues with int and numpy that was deprecated, so either use more recent version or manually edit 
so that `np.int` or `np.float` is `int` and `float`.
```bash
criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

##########################################################
criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,

            # weight_dict=weight_dict, This did not work...
            class_weight=class_weight,
            mask_weight=mask_weight,
            dice_weight=dice_weight,

            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

```
Changes made before installation: 

`detectron2` has some issues with int and numpy that was deprecated. Any reference to `np.int` or `np.float` can be
replaced with `int` and `float`, this will not produce a warning message and will have the same behavior.

Files added from original repository, please change `_root` to the directory where perturbations are stored:
`models/Mask2Former/mask2former/data/datasets/register_ROBUSTNESS_coco_panoptic_annos_semseg.py` and same for
`models/Mask2Former/mask2former/data/datasets/register_ROBUSTNESS_ade20k_panoptic.py`

added `register_ROBUSTNESS_coco_panoptic_annos_semseg`  to `models/Mask2Former/mask2former/data/datasets/__init__.py`

added `register_ROBUSTNESS_ade20k_panoptic` to `models/Mask2Former/mask2former/data/datasets/__init__.py`
```bash
conda create --name mask2former python=3.8 -y
source activate mask2former
module load cuda/11.7

conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.7 -c pytorch -c nvidia
 /home/schiappa/.conda/envs/mask2former/bin/pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

```

To update datasets:

```bash
python register_<new dataset file>.py
```
## Running
Model types available are:

For COCO: 
* r50_panoptic
* r50_instance
* swinL_panoptic
* swinL_instance

For ADE20K:
* r50_panoptic
* r50_instance
* r50_semantic
* swinL_instance
* swinL_panoptic
* swinL_semantic

To run call, an example:
```bash
conda avtivate mask2former
python eval_mask2former.py --model_type <model_type> --corruption <corruption>--severity <severity> --dataset <dataset>
```


# MaskedDino
"Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation"
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mask-dino-towards-a-unified-transformer-based-1/panoptic-segmentation-on-coco-minival)](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-minival?p=mask-dino-towards-a-unified-transformer-based-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mask-dino-towards-a-unified-transformer-based-1/panoptic-segmentation-on-coco-test-dev)](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-test-dev?p=mask-dino-towards-a-unified-transformer-based-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mask-dino-towards-a-unified-transformer-based-1/instance-segmentation-on-coco-minival)](https://paperswithcode.com/sota/instance-segmentation-on-coco-minival?p=mask-dino-towards-a-unified-transformer-based-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mask-dino-towards-a-unified-transformer-based-1/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=mask-dino-towards-a-unified-transformer-based-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mask-dino-towards-a-unified-transformer-based-1/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=mask-dino-towards-a-unified-transformer-based-1)


[Feng Li*](https://fengli-ust.github.io/), [Hao Zhang*](https://scholar.google.com/citations?user=B8hPxMQAAAAJ&hl=zh-CN), [Huaizhe Xu](https://scholar.google.com/citations?user=zgaTShsAAAAJ&hl=en&scioq=Huaizhe+Xu), [Shilong Liu](https://www.lsl.zone/), [Lei Zhang](https://scholar.google.com/citations?hl=zh-CN&user=fIlGZToAAAAJ), [Lionel M. Ni](https://scholar.google.com/citations?hl=zh-CN&user=OzMYwDIAAAAJ), and [Heung-Yeung Shum](https://scholar.google.com.hk/citations?user=9akH-n8AAAAJ&hl=en).

## Installation
Original installation instructions is in [models/MaskedDino/Install.md](https://github.com/IDEA-Research/MaskDINO/blob/main/INSTALL.md).

Files added from original repository, please change `_root` to the directory where perturbations are stored:
`models/MaskDino/maskdino/data/datasets/register_ROBUSTNESS_coco_panoptic_annos_semseg.py`

added `register_ROBUSTNESS_coco_panoptic_annos_semseg`  to `models/MaskDino/maskdino/data/datasets/__init__.py`

```bash
conda create --name maskdino python=3.8 -y
source activate maskdino
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

git clone https://github.com/IDEA-Research/MaskDINO.git
cd MaskDINO
pip install -r requirements.txt
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
```
## Running
Model types available are:
For COCO:
* r50_panoptic
* swinL_panoptic

For ADE20K:
* r50_semantic

To run call, an example:
```bash
conda avtivate maskdino
python eval_maskdino.py --model_type <model_type> --corruption <corruption>--severity <severity> --dataset <dataset>
```