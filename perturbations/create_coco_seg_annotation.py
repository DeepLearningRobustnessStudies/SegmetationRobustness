import functools
import json
import multiprocessing as mp
import numpy as np
import os
import time
from fvcore.common.download import download
from panopticapi.utils import rgb2id
from PIL import Image

from detectron2.builtin_meta import COCO_CATEGORIES

# root_dataset_path( which contains all the perturbed dataset)
dataset_dir = os.path.join("/home/c3-0", "datasets", "robustness_coco")
noise_type = ["rotate","translate","shear"] 
severity = [1,2,3,4,5]

def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = new_cat_id
    Image.fromarray(output).save(output_semantic)


def separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.
    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.
    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format.
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(categories) <= 254
    for i, k in enumerate(categories):
        id_map[k["id"]] = i
    # what is id = 0?
    # id_map[0] = 255
    print(id_map)

    with open(panoptic_json) as f:
        obj = json.load(f)

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for anno in obj["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name)
            output = os.path.join(sem_seg_root, file_name)
            yield input, output, segments

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))



for noise in noise_type:
    for sev in severity:
        for s in ["val2017"]:
            data_dir = os.path.join(dataset_dir,noise,str(sev),"coco")
            separate_coco_semantic_from_panoptic(
                os.path.join(data_dir, "annotations/panoptic_{}.json".format(s)),
                os.path.join(data_dir, "panoptic_{}".format(s)),
                os.path.join(data_dir, "panoptic_semseg_{}".format(s)),
                COCO_CATEGORIES,
            )


import json
import os
from collections import defaultdict


def load_coco_caption():
    id2caption = defaultdict(list)
    dataset_dir = os.path.join("/home/c3-0", "datasets", "coco")
    for json_file in ["captions_val2017.json"]:
        with open(os.path.join(dataset_dir, "annotations", json_file)) as f:
            obj = json.load(f)
            for ann in obj["annotations"]:
                id2caption[int(ann["image_id"])].append(ann["caption"])

    return id2caption


def create_annotation_with_caption(input_json, output_json):
    id2coco_caption = load_coco_caption()

    with open(input_json) as f:
        obj = json.load(f)

    coco_count = 0

    print(f"Starting to add captions to {input_json} ...")
    print(f"Total images: {len(obj['annotations'])}")
    for ann in obj["annotations"]:
        image_id = int(ann["image_id"])
        if image_id in id2coco_caption:
            ann["coco_captions"] = id2coco_caption[image_id]
            coco_count += 1
    print(f"Found {coco_count} captions from COCO ")

    print(f"Start writing to {output_json} ...")
    with open(output_json, "w") as f:
        json.dump(obj, f)


dataset_dir = os.path.join("/home/c3-0", "datasets", "robustness_coco")
noise_type = ["rotate","translate","shear"] 
severity = [1,2,3,4,5]

for noise in noise_type:
    for sev in severity:
        for s in ["val2017"]:
            data_dir = os.path.join(dataset_dir,noise,str(sev),"coco")
            create_annotation_with_caption(
                os.path.join(data_dir, "annotations/panoptic_{}.json".format(s)),
                os.path.join(data_dir, "annotations/panoptic_caption_{}.json".format(s)),
            )
            