# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import itertools

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES

import pdb
import os
import sys
import json
from pathlib import Path
from odise.data import get_openseg_labels

import argparse
import logging
import os.path as osp
from contextlib import ExitStack
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import create_ddp_model, default_argument_parser, hooks, launch
from detectron2.engine import create_ddp_model, default_argument_parser, hooks, launch
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from iopath.common.s3 import S3PathHandler

from odise.checkpoint import ODISECheckpointer
from odise.config import auto_scale_workers, instantiate_odise
from odise.engine.defaults import default_setup

from odise.utils.events import CommonMetricPrinter, WandbWriter, WriterStack
from models.ODISE.tools.train_net import default_writers, do_test
from utils.process_results import post_process_results_detectron2

# from models.ODISE.third_party.Mask2Former.mask2former.data.datasets.register_ROBUSTNESS_ade20k_panoptic import *
# PathManager.register_handler(S3PathHandler())
# setup_logger()
# logger = setup_logger(name="odise")


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg.train.run_name = (f"{args.model_type}_{args.corruption}_{args.severity}")

    if hasattr(args, "reference_world_size") and args.reference_world_size:
        cfg.train.reference_world_size = args.reference_world_size
    cfg = auto_scale_workers(cfg, comm.get_world_size())
    cfg.train.cfg_name = osp.splitext(osp.basename(args.config_file))[0]
    if hasattr(args, "output") and args.output:
        cfg.train.output_dir = args.output
    else:
        cfg.train.output_dir = osp.join("output", cfg.train.run_name)
    if hasattr(args, "tag") and args.tag:
        cfg.train.run_tag = args.tag
        cfg.train.output_dir = osp.join(cfg.train.output_dir, cfg.train.run_tag)
    if hasattr(args, "wandb") and args.wandb:
        cfg.train.wandb.enable_writer = args.wandb
        cfg.train.wandb.enable_visualizer = args.wandb
    if hasattr(args, "amp") and args.amp:
        cfg.train.amp.enabled = args.amp
    if hasattr(args, "init_from") and args.init_from:
        cfg.train.init_checkpoint = args.init_from
    cfg.train.log_dir = cfg.train.output_dir
    if hasattr(args, "log_tag") and args.log_tag:
        cfg.train.log_dir = osp.join(cfg.train.log_dir, args.log_tag)

    if 'model_type' in args.opts:
        setattr(args, 'opts', [])
    if args.dataset == 'coco':
        cfg.dataloader.test.dataset.names = f'coco_2017_val_panoptic_{args.corruption}_{args.severity}_with_sem_seg'
    elif args.dataset == 'ade20k':
        cfg.dataloader.test.dataset.names = f'ade20k_panoptic_val_{args.corruption}_{args.severity}'
    else:
        raise NotImplementedError

    # this information is pretty much usefuless, so not colelcting now
    # for i in range(len(cfg.dataloader.evaluator)):
    #     cfg.dataloader.evaluator[i]['output_dir'] = os.path.join(args.output, 'evaluators')

    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    logger = setup_logger(cfg.train.log_dir, distributed_rank=comm.get_rank(), name="odise")
    logger.info(f"Changed test dataset name to {cfg.dataloader.test.dataset.names}")
    logger.info(f"Running with config:\n{LazyConfig.to_py(cfg)}")

    model = instantiate_odise(cfg.model)
    # # TEMPORARY
    # cfg.train.device = 'cpu'
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    ODISECheckpointer(model, cfg.train.output_dir).resume_or_load(
        cfg.train.init_checkpoint, resume=args.resume
    )

    with ExitStack() as stack:
        stack.enter_context(
            WriterStack(
                logger=logger,
                writers=default_writers(cfg) if comm.is_main_process() else None,
            )
        )
        results = do_test(cfg, model, final_iter=True)
        logger.info(results)

        with open(os.path.join(args.output, 'results.json'), 'w') as f:
            json.dump(results, f)
    # Evaluation may take different time among workers.
    # A barrier make them start the next iteration together.
    comm.synchronize()
    post_process_results_detectron2(args, results)



if __name__ == '__main__':
    # Original from `tools/train_net.py`
    parser = argparse.ArgumentParser(
        "odise training and evaluation script",
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
    parser.add_argument('--model_type', default='label', type=str, help="Options are label or caption")
    parser.add_argument('--corruption', default='clean', type=str, help="Have somewhere that lists corruptions")
    parser.add_argument('--severity', default=0, type=int,
                        help="Severity of the corruption. If CLEAN is passed, not applicable")
    parser.add_argument('--dataset', default='coco')
    parser.add_argument("--save_dir", default=f"/home/c3-0/datasets/robustness/lvlm_robustness/image_domain/",
                        help="Directory to save post-processed results for further analysis.")

    args = parser.parse_args()

    # Get the model type we want to eval on
    if args.model_type == 'label':
        model_name = "ODISE(Label)"
        args.init_from = 'weights/odise_label_coco_50e-b67d2efc.pth'
        if args.dataset == 'coco':
            args.config_file = "models/ODISE/configs/Panoptic/odise_label_coco_ONLY_50e.py"
        elif args.dataset == 'ade20k':
            args.config_file = 'models/ODISE/configs/Panoptic/odise_label_ade20k_50e.py'
    elif args.model_type == 'caption':
        model_name = "ODISE(Caption)"
        args.init_from = 'weights/odise_caption_coco_50e-853cc971.pth'
        if args.dataset == 'coco':
            args.config_file = "models/ODISE/configs/Panoptic/odise_caption_coco_ONLY_50e.py"
        elif args.dataset == 'ade20k':
            args.config_file = "models/ODISE/configs/Panoptic/odise_caption_adek20_50e.py"

    output_dir = os.path.join(args.output, f'{args.dataset.upper()}_ODISE_{args.model_type}', f"{args.corruption}_{args.severity}")
    args.save_dir = os.path.join(args.save_dir, args.dataset, f'ODISE_{args.model_type}')

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

    # Print arguments
    logger.info(f"Storing log path in {log_path}")
    logger.info("Model configurations")
    logger.info('-------------------------')
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


# CHECK VALIDITY OF IMAGES
# for filename in tqdm.tqdm(images, total=len(images)):
#     try:
#         im = Image.open(filename)
#         im.verify() #I perform also verify, don't know if he sees other types o defects
#         im.close() #reload is necessary in my case
#         im = Image.open(filename)
#         _ = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
#         numpy_im = np.asarray(im)
#         im.close()
#     except Exception as e:
#         print(f"{e}: {filename}")