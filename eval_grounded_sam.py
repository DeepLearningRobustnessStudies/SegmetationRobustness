import argparse
import os
import copy

import numpy as np
import json
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import logging
import sys
import cv2
import pdb
from copy import copy

print("Importing pycocotools...")
from pycocotools.mask import encode
from pycocotools.coco import COCO

print("Importing torch...")
import torch
import torchvision
from torch.utils.data import Dataset

# Grounding DINO
print("Importing GroundedSAM..")
from models.GroundedSAM.GroundingDINO.groundingdino.models import build_model
from models.GroundedSAM.GroundingDINO.groundingdino.util.slconfig import SLConfig
import models.GroundedSAM.GroundingDINO.groundingdino.datasets.transforms as T
from models.GroundedSAM.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
print("Importing segment anything...")
from segment_anything import build_sam, SamPredictor

print("Importing huggingface_hub")
from huggingface_hub import hf_hub_download

print("Importing local files...")
from utils.process_results import post_process_results_detectron2
from utils.coco_eval import CocoDataset1
from utils.ade_eval import ADE20KDataset

import warnings
warnings.filterwarnings("ignore")


class RobustnessImages(Dataset):
    def __init__(self, args, dataset, split='. ', query=10):
        """
        Dataset is from detectron2 and ODISE.
        It has [0, 132] category IDs. My guess is no ignore index?
        COCO boxes are in  [x,y,w,h]  format
        Current boxes are set up for size (426, 640), output of ODISE
        Args:
            args:
            dataset:
            split:
            query:
        """
        self.queries = query
        self.split = split
        self.dataset = dataset
        old_dims = (426, 640)
        new_dims = (480, 640)
        self.image_ids = list(self.dataset.keys())
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def resize_boxes(bboxes, new_dims, old_dims):
        """
        Validated with:
            from utils.visualization import show_bbox
            import torchvision
            import matplotlib.pyplot as plt
            resize_1 = torchvision.transforms.Resize(old_dims)
            resize_2 = torchvision.transforms.Resize(new_dims)
            bboxes_orig = torch.tensor([torch.tensor(x['bbox'] for x in inputs[0]['segments_info']])
            show_bbox(resize_1(sample_img), bboxes_orig); plt.show()
            show_bbox(resize_2(sample_img), bboxes); plt.show()

        COCO bbox in format: [x,y,w,h]
        Args:
            bboxes:

        Returns:

        """
        inp_w, inp_h = new_dims[1], new_dims[0]
        w, h = old_dims[1], old_dims[0]

        scale_w = inp_w / w
        scale_h = inp_h / h

        bboxes[:, 0] *= (scale_w)
        bboxes[:, 2] *= (scale_w)
        bboxes[:, 1] *= (scale_h)
        bboxes[:, 3] *= (scale_h)


        new_w = scale_w * w
        new_h = scale_h * h

        del_h = (inp_h - new_h) / 2
        del_w = (inp_w - new_w) / 2

        add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)

        bboxes[:, :4] += add_matrix

        return bboxes

    def __getitem__(self, idx):
        """
        Original Code:
        image non-normalized image of (3, 800, 1201)  -> Passed to GroundedDINO
        image_source normalized image of (426, 640, 3) -> Passed to SAM

        Detectron2 Code:
        image non-normalized image of shape [3, 1024, 1538] my guess is makes this bigger
        sem_seg non-normalized image of [1024, 1538]
        Has bounding box info, but should not be using it so should be fine.

        Args:
            idx:

        Returns:

        """
        image_id = self.image_ids[idx]
        image_path = os.path.join(args.root_image_dir, args.corruption, args.severity, 'coco', 'val2017', self.dataset[image_id])

        # For grounding DINO
        _, image = self.load_image(image_path)

        # For sam predictor
        image_source = cv2.imread(image_path)
        image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)

        return image_source, image, image_id


    def load_image(self, image_path):
        transform = self.transform
        image_source = Image.open(image_path).convert("RGB")   #-> Reads image as 640x426x3
        # image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image_source, image_transformed


def build_dataset_and_eval(args):
    if args.dataset == 'coco':
        dataset_path = os.path.join(args.root_image_dir, args.corruption, args.severity, 'coco')
        inst_ann_path = os.path.join(dataset_path, "annotations", "instances_val2017.json")
        evaluator = CocoDataset1
    elif args.dataset == 'ade20k':
        dataset_path = os.path.join(args.root_image_dir, args.corruption, args.severity, 'ade/ADEChallengeData2016')
        inst_ann_path = os.path.join(dataset_path, "ade20k_instance_val.json")
        evaluator = ADE20KDataset
    else:
        raise NotImplementedError

    # img_dataset_path = os.path.join(dataset_path, "val2017")
    coco_ann = COCO(inst_ann_path)

    categories = coco_ann.cats

    class_mappings = {}
    for i in categories.keys():
        class_mappings[categories[i]['name'].replace(" ", "")] = i

    # get id_file_name dict
    id_dict = {}
    for ann in coco_ann.dataset["images"]:
        id_dict[ann["id"]] = ann["file_name"]

    dataset = RobustnessImages(args, id_dict)

    # transform = [
    #     T.ToTensor(),
    #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ]
    transform = [T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 ]
    evaluator = evaluator(ann_file=inst_ann_path, pipeline=transform)

    return dataset, evaluator, class_mappings


def run_eval(args, groundingdino_model, sam_predictor):
    """
    list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the
                        values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
                * "pred_boxes":
                     A dictionary that has "boxes", "scores", "pred_classes"
                     Boxes contains a detectron2.structures.Boxes object
                     scores is a tensor  of logit scores 1D
                     pred_classes is a tensor of 1D of classes
    Args:
        args:
        groundingdino_model:
        sam_predictor:

    Returns:

    """
    dataset, evaluator, class_mappings = build_dataset_and_eval(args)

    device = args.device

    coco_grounded = COCOGrounded(args, groundingdino_model, sam_predictor, class_mappings)

    pbar = tqdm(dataset, total=len(dataset), position=0, leave=True)
    bbox_results = []
    phrases_results = []
    masks_results = []
    img_id_list = []
    scores = []
    for image_source, image, image_id in pbar:
        image = image.to(device)

        boxes, logits, classes  = coco_grounded.get_grounding_dino(image)
        if (len(boxes) > 0):
            masks, iou_predictions, low_res_masks,boxes_filt = coco_grounded.get_sam(image_source, boxes)
            masks = coco_grounded.preprocess_outputs(masks)
        else:
            masks = list()

        # Collect
        bbox_results.append(boxes.cpu())
        phrases_results.append(classes)
        masks_results.append(masks)
        img_id_list.append(image_id)
        scores.append(logits.cpu())

    results = [(bbox_results, phrases_results, scores, masks_results, img_id_list, class_mappings)]
    results, save_results = evaluator.evaluate(results, "segm", classwise=True)

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f)


class COCOGrounded(object):
    def __init__(self, args, groundingdino_model, sam_predictor, class_mappings):
        self.device = args.device
        self.groundingdino_model = groundingdino_model.to(args.device)
        self.sam_predictor = sam_predictor
        self.box_threshold = args.box_threshold
        self.text_threshold = args.text_threshold
        self.class_mappings = class_mappings
        self.categories = list(self.class_mappings.keys())
        self.max_text_len = 256
        self.tokenizer = self.groundingdino_model.tokenizer
        self.tokenizer_class_mappings = {cls: self.tokenizer(cls)['input_ids'][1:-1] for cls in self.class_mappings.keys()}
        self.captions =  self.get_captions()

    def get_sam(self, image, boxes):

        with torch.no_grad():
            self.sam_predictor.set_image(image)
            H, W, _ = image.shape
            # boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H]).to(self.device)
            # transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(self.device)
            for i in range(boxes.size(0)):
                boxes[i] = boxes[i] * torch.Tensor([W, H, W, H]).to(self.device)
                boxes[i][:2] -= boxes[i][2:] / 2
                boxes[i][2:] += boxes[i][:2]
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2]).to(self.device)

            masks, iou_predictions, low_res_masks = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

        return masks, iou_predictions, low_res_masks, boxes

    def get_grounding_dino(self, image):
        image = image.to(self.device)

        logits, boxes, classes = list(), list(), list()

        for new_caption in self.captions:

            with torch.no_grad():
                outputs = self.groundingdino_model(image[None], captions=[new_caption])

            chunk_classes, chunk_logits, chunk_boxes = self.get_logits_per_class(new_caption, outputs)
            logits.append(chunk_logits)
            boxes.append(chunk_boxes)
            classes.append(chunk_classes)

        logits = torch.cat(logits, dim=0)
        boxes = torch.cat(boxes, dim=0)

        classes = list(np.concatenate(classes))
        return boxes, logits, classes

    def get_captions(self):
        tokens = list()
        classes = list()
        captions = list()
        for idx, (cls, tok) in enumerate(self.tokenizer_class_mappings.items()):
            tokens += tok + [1012]
            classes.append(cls)
            if len(tokens) >= self.max_text_len - 2 or idx +1 == len(self.tokenizer_class_mappings):
                captions.append('. '.join(classes) +'.')
                tokens = list()
                classes = list()
        return captions

    def get_logits_per_class(self, new_caption, outputs):
        """
        Will return the (N masks, Logits per class for each mask)
        Since we are sliding window across all classes, we will use 0 for those that are not present.
        This is a hack but ?
        Args:
            new_caption:
            outputs:

        Returns:

        """
        prediction_logits = outputs["pred_logits"].sigmoid()[0]  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"][0]

        mask = prediction_logits.max(dim=1)[0] > self.box_threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        # Modified how to select phrases
        tokenized = self.tokenizer(new_caption)
        tokens = np.array(tokenized['input_ids'])

        # We need to pick the top class based on tokens.
        logit_dictionary = dict()
        for idx, (cls, cls_tokens) in enumerate(self.tokenizer_class_mappings.items()):
            indices = list()
            for tok in cls_tokens:
                if tok in tokens:
                    indices.append(np.where(tokens == tok)[0][0])
            if len(indices) > 0:
                # Max is to that we can pick the maximum logit between a word that has multiple tokens
                logit_dictionary[idx] = torch.max(logits[:, indices], dim=-1)[0]
        cls_logits, classes = torch.max(torch.stack(list(logit_dictionary.values())), dim=0)

        # Cause want word level represenations
        phrases = [self.categories[cls.item()] for cls in classes]

        return phrases, cls_logits, boxes

    @staticmethod
    def preprocess_outputs(masks):
        masks = masks.squeeze(1).cpu().numpy()
        processed_mask = []
        if (len(masks) == 0):
            return processed_mask

        for mask in masks:
            mask = encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            processed_mask.append(mask)
        return processed_mask


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model



if __name__ == "__main__":

    parser = argparse.ArgumentParser("GroundedSAM Robustness", add_help=True)
    parser.add_argument(
        "--output_dir", "-o", type=str, default="output", help="output directory"
    )
    parser.add_argument(
        "--output", type=str, default="output", help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")

    parser.add_argument('--dataset', default='coco', type=str)
    parser.add_argument('--corruption', default='clean', type=str)
    parser.add_argument('--severity', default='0', type=str)
    parser.add_argument('--model_type', default='large', type=str)
    parser.add_argument("--save_dir", default=f"/home/c3-0/datasets/robustness/lvlm_robustness/image_domain/",
                        help="Directory to save post-processed results for further analysis.")
    parser.add_argument('--root_image_dir', default='/home/c3-0/datasets/robustness_coco', type=str)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, f'{args.dataset.upper()}_GroundedSAM_{args.model_type}',
                              f"{args.corruption}_{args.severity}")
    args.save_dir = os.path.join(args.save_dir, args.dataset, f'GroundedSAM_{args.model_type}')

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

    # cfg
    if args.model_type == 'swinT':
        logger.info(f"Loading model with SwinT as GroundingDINO backbone from lcoal files.")
        config_file='models/GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        grounded_checkpoint = 'weights/groundingdino_swint_ogc.pth'
        sam_checkpoint = 'weights/sam_vit_h_4b8939.pth'
        hugging_face = False
        model = load_model(config_file, grounded_checkpoint, device=args.device)
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(args.device))
        args.model_type = 'swinT'

    elif args.model_type == 'swinB':
        logger.info(f"Loading model with SwinB as GroundingDINO backbone from huggingface.")
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        grounded_checkpoint = "groundingdino_swinb_cogcoor.pth"
        confg_file = "GroundingDINO_SwinB.cfg.py"
        sam_checkpoint= 'weights/sam_vit_h_4b8939.pth'
        hugging_face = True
        model = load_model_hf(ckpt_repo_id, grounded_checkpoint, confg_file, device=args.device)
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device=args.device)
        predictor = SamPredictor(sam)
    else:
        logger.error(f"Please pass valid model, either `swinB` or `swinT`")
        raise NotImplementedError

    assert args.dataset in ['coco', 'ade20k'], f"Invalid dataset passed {args.dataset}, pass one of `coco` or `ade20k`"

    run_eval(args, model, predictor)
