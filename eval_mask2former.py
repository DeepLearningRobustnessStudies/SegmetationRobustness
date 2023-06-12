 # python eval_mask2former_coco.py \
 #  --config-file /home/schiappa/PanopticRobustness/models/ODISE/third_party/Mask2Former/configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
 # --eval-only MODEL.WEIGHTS /path/to/checkpoint_file

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.

This script is modified from the original repository: https://github.com/facebookresearch/Mask2Former
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import json
import pdb
import copy
import itertools
import logging
import os
import argparse
import sys
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from models.Mask2Former.mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

from utils.process_results import post_process_results_detectron2
"""
        if task_type == 'segm':
            overall_metric = 'AP'
            object_metric = 'AP-'
            new_results = collect(new_results, res, overall_metric, object_metric)
        elif task_type == "sem_seg":
            overall_metric = 'mIoU'
            object_metric = 'IoU-'
            new_results = collect(new_results, res, overall_metric, object_metric)
            overall_metric = 'mACC'
            object_metric = 'ACC-'
            new_results = collect(new_results, res, overall_metric, object_metric)
        else:
            overall_metric = 'PQ'
            object_metric = None
            new_results = collect(new_results, res, overall_metric, object_metric)

"""
TASK_MAPPING = {
    "segm": [{'overall_metric': 'AP', 'object_metric': 'AP-'}],
    "sem_seg": [{"overall_metric": "mIoU", 'object_metric': "IoU-"},
                {"overall_metric": 'mACC', 'object_metric': 'ACC-'}],
    'panoptic_seg': [{'overall_metric': 'PQ', 'object_metric': None},
                     {'overall_metric': 'SQ', 'object_metric': None},
                     {'overall_metric': 'RQ', 'object_metric': None}],

}


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.dataset == 'coco':
        cfg.DATASETS.TEST = (f'coco_2017_val_panoptic_{args.corruption}_{args.severity}_with_sem_seg',)
    elif args.dataset == 'ade20k':
        cfg.DATASETS.TEST = (f'ade20k_panoptic_val_{args.corruption}_{args.severity}',)

    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)
    # cfg.defrost()
    # cfg.DATASETS.TEST = (f'coco_2017_val_panoptic_{args.corruption}_{args.severity}_with_sem_seg',)
    # cfg.MODEL.WEIGHTS = args.model_weights
    # cfg.freeze()

    args.eval_only = True
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    res = Trainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
    post_process_results_detectron2(args, res)
    return res




def validate_results_already_exists(args):
    save_dir = f"/home/c3-0/datasets/robustness/lvlm_robustness/image_domain/Mask2Former_{args.model_type}/"
    corruption = args.corruption
    severity = str(args.severity)

    save_dir = os.path.join(save_dir, corruption, severity)
    if os.path.isfile(os.path.join(save_dir, 'results.json')):
        return True
    else:
        return False


if __name__ == "__main__":
    # Original from `tools/train_net.py`
    parser = argparse.ArgumentParser(
        "mask2former evaluation script",
        parents=[default_argument_parser()],
        add_help=False,
    )

    parser.add_argument(
        "--output", default='output',
        type=str,
        help="root of output folder, " "the full path is <output>/<model_name>/<tag>",
    )
    parser.add_argument("--init-from", type=str, help="init from the given checkpoint")
    parser.add_argument("--tag", default="default", type=str, help="tag of experiment")
    parser.add_argument("--log-tag", type=str, help="tag of experiment")
    parser.add_argument("--wandb", action="store_true", help="Use W&B to log experiments")
    parser.add_argument("--amp", action="store_true", help="Use AMP for mixed precision training")
    parser.add_argument("--reference-world-size", "--ref", type=int)

    # Added
    parser.add_argument('--model_type', default='r50_panoptic', type=str, help="Options are label or caption")
    parser.add_argument('--corruption', default='clean', type=str, help="Have somewhere that lists corruptions")
    parser.add_argument('--severity', default=0, type=int,
                        help="Severity of the corruption. If CLEAN is passed, not applicable")
    parser.add_argument('--model_weights', type=str)
    parser.add_argument('--dataset', default='coco', type=str)
    parser.add_argument("--save_dir", default=f"/home/c3-0/datasets/robustness/lvlm_robustness/image_domain/",
                        help="Directory to save post-processed results for further analysis.")
    parser.add_argument('--root_dir', default='/home/schiappa/PanopticRobustness/', type=str)

    args = parser.parse_args()

    output_dir = os.path.join(args.output, f'{args.dataset.upper()}_Mask2Former_{args.model_type}', f"{args.corruption}_{args.severity}")
    args.save_dir = os.path.join(args.save_dir, args.dataset, f'Mask2Former_{args.model_type}')

    log_path = os.path.join(output_dir, 'log.txt')
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True)

    if os.path.isfile(log_path) and os.path.isfile(os.path.join(output_dir, 'log.pth')):
        print(f"Model already exists. Exiting...")
        sys.exit()

    args.output = output_dir

    # Create a Logger Object - Which listens to everything
    logger = logging.getLogger(os.path.basename(__file__))
    logger.setLevel(logging.DEBUG)

    # Register the Console as a handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Log format includes date and time
    formatter = logging.Formatter('%(asctime)s %(levelname)-5s %(message)s')
    ch.setFormatter(formatter)

    # If want to print output to screen
    logger.addHandler(ch)

    # Create a File Handler to listen to everything
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)

    # Log format includes date and time
    fh.setFormatter(formatter)

    # Register it as a listener
    logger.addHandler(fh)

    ROOT_DIR = args.root_dir

    # Get the model type we want to eval on
    if args.dataset == 'coco':
        if args.model_type == 'r50_panoptic':
            args.config_file = "models/ODISE/third_party/Mask2Former/configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml"
            args.model_weights = ROOT_DIR + 'weights/mask2former_panoptic_coco_r50_model_final_94dc52.pkl'
        elif args.model_type == 'r50_instance':
            args.config_file = "models/ODISE/third_party/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
            args.model_weights = ROOT_DIR + 'weights/mask2former_instance_coco_r50_model_final_3c8ec9.pkl'
        elif args.model_type == "swinL_panoptic":
            args.config_file = "models/ODISE/third_party/Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
            args.model_weights = ROOT_DIR + "weights/mask2former_panoptic_coco_swinL_model_final_f07440.pkl"
        elif args.model_type == 'swinL_instance':
            args.config_file = "models/ODISE/third_party/Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
            args.model_weights = ROOT_DIR + "weights/mask2former_instance_coco_swinL_model_final_e5f453.pkl"
        else:
            logger.error(f"Passed invalid model_type {args.model_type}")
            sys.exit()
    elif args.dataset == 'ade20k':
        if args.model_type == 'r50_panoptic':
            args.model_weights = ROOT_DIR + 'weights/mask2former_panoptic_ade20k_r50_model_final_5c90d4.pkl'
            args.config_file = 'models/ODISE/third_party/Mask2Former/configs/ade20k/panoptic-segmentation/maskformer2_R50_bs16_160k.yaml'
        elif args.model_type == 'r50_instance':
            args.model_weights = ROOT_DIR + 'weights/mask2former_instance_ade20k_r50_model_final_67e945.pkl'
            args.config_file = 'models/ODISE/third_party/Mask2Former/configs/ade20k/instance-segmentation/maskformer2_R50_bs16_160k.yaml'
        elif args.model_type == 'r50_semantic':
            args.model_weights = ROOT_DIR + 'weights/mask2former_semantic_ade20k_r50_model_final_500878.pkl'
            args.config_file = 'models/ODISE/third_party/Mask2Former/configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml'
        elif args.model_type == 'swinL_instance':
            args.model_weights = ROOT_DIR + 'weights/mask2former_instance_ade20k_swinL_model_final_92dae9.pkl'
            args.config_file = 'models/ODISE/third_party/Mask2Former/configs/ade20k/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml'
        elif args.model_type == 'swinL_panoptic':
            args.model_weights = ROOT_DIR + 'weights/mask2former_panoptic_ade20k_swinL_model_final_e0c58e.pkl'
            args.config_file = 'models/ODISE/third_party/Mask2Former/configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml'
        elif args.model_type == 'swinL_semantic':
            args.model_weights = ROOT_DIR + 'weights/mask2former_semantic_ade20k_swinL_model_final_6b4a3a.pkl'
            args.config_file = 'models/ODISE/third_party/Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml'
    else:
        logger.error(f"Passed invalid dataset, should be either `coco` or `ade20k`")

    # Print arguments
    logger.info(f"Storing log path in {log_path}")
    logger.info("Model configurations")
    logger.info('-------------------------')
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # if validate_results_already_exists(args):
    #     logger.info(f"Results already exist for {args.model_type} for {args.corruption} at severity {args.severity}. "
    #                 f"Delete if want to run again.")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
