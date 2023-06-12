import os
import shutil
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from perturb import *
import json
from PIL import Image
from multiprocessing import Pool, Process, Manager, Value
from pycocotools.coco import COCO
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="generate perturbed coco dataset")
    parser.add_argument("--coco_path", default="/home/c3-0/datasets/coco/", help="path to coco dataset")
    parser.add_argument("--save_dir", default="/home/c3-0/datasets/robustness_coco/", help="path to save the perturbed dataset")

    return parser


args = get_parser().parse_args()

dataset_path = args.coco_path
save_dir = args.save_dir

dataset_val_path = os.path.join(dataset_path, "val2017")
save_val_dir = "coco/val2017"

annotation_path = os.path.join(dataset_path, "annotations")
caption_ann_path = os.path.join(annotation_path, "captions_2017val.json")

noise_types = ["gaussian","shot","impulse","speckle","defocus","motion","zoom","contrast","brightness","darkness","jpeg","pixelate","fog","snow","rotate","translate","shear"]
severity = [1,2,3,4,5]

# load the annotations
instance_annotations = json.load(open(annotation_path + "instances_val2017.json","r"))
coco_ann = COCO(annotation_path + "instances_val2017.json")
panoptic_ann = json.load(open(annotation_path + "panoptic_val2017.json","r"))


id_dict = {}
for ann in instance_annotations["images"]:
    id_dict[ann["id"]] = ann["file_name"]

img_id_list = list(id_dict.keys())
img_name_list = list(id_dict.values())
print(len(img_name_list))
# img_name_list = img_name_list[:5]
img_d_correspondence = {}
for i in range(len(img_name_list)):
    img_d_correspondence[img_name_list[i]] = np.random.choice([-1,1],2)

def polygonFromMask(maskedArr):
  # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
  contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  segmentation = []
  area = cv2.countNonZero(maskedArr)
  bbox = cv2.boundingRect(cv2.findNonZero(maskedArr))
  valid_poly = 0
  for contour in contours:
  # Valid polygons have >= 6 coordinates (3 points)
     if contour.size >= 6:
        segmentation.append(contour.astype(float).flatten().tolist())
        valid_poly += 1
#   if valid_poly == 0:
#     plt.imshow(maskedArr)
#     plt.show()
#     raise ValueError
  return segmentation, area, bbox, valid_poly


def generate_perturbed_images(noise_type, severity):
    for noise in noise_type:
        for sev in severity:
            # Overwrite the modified annotations to original annotations
            instance_annotations = json.load(open(annotation_path + "instances_val2017.json","r"))
            coco_ann = COCO(annotation_path + "instances_val2017.json")
            panoptic_ann = json.load(open(annotation_path + "panoptic_val2017.json","r"))
            
            if(noise not in ["translate","rotate","shear","dropout"]):
                # Erase the previous data annotations
                if(os.path.exists(os.path.join(save_dir,noise,str(sev),"coco"))):
                    shutil.rmtree(os.path.join(save_dir,noise,str(sev),"coco"))
                
                shutil.copytree(annotation_path,os.path.join(save_dir,noise,str(sev),"coco/annotations"))
                shutil.copytree(dataset_path + "panoptic_val2017",os.path.join(save_dir,noise,str(sev),"coco/panoptic_val2017"))
                shutil.copytree(dataset_path + "panoptic_semseg_val2017",os.path.join(save_dir,noise,str(sev),"coco/panoptic_semseg_val2017"))

                perturb = ImagePerturbation(noise, sev=sev).perturb
                print("Starting to generate perturbed images for {} with severity {}".format(noise,sev))
                for file_name in img_name_list:
                    img = cv2.imread(os.path.join(dataset_val_path,file_name))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = torch.from_numpy(img).float()
                    perturb = ImagePerturbation(noise, sev=sev)
                    img_perturb, _ = perturb(img, None)
                    # if img is in pil format then convert it to numpy array
                    if isinstance(img_perturb, Image.Image):
                        img_perturb = np.array(img_perturb)
                    img_perturb = cv2.cvtColor(img_perturb, cv2.COLOR_RGB2BGR)
                    save_path = os.path.join(save_dir,noise,str(sev),save_val_dir)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path,file_name),img_perturb)
            
            else:
                inst_save_id = {}
                panop_save_id = {}
                for i in range(len(img_name_list)):
                    inst_save_id[img_name_list[i]] = 0
                    panop_save_id[img_name_list[i]] = 0

                ann_list = [0]*len(instance_annotations["annotations"])
                # Erase the previous data annotations
                if(os.path.exists(os.path.join(save_dir,noise,str(sev),"coco"))):
                    shutil.rmtree(os.path.join(save_dir,noise,str(sev),"coco"))
                
                if(os.path.exists(os.path.join(save_dir,noise,str(sev),"coco/annotations"))==False):
                    os.makedirs(os.path.join(save_dir,noise,str(sev),"coco/annotations"))
                shutil.copy(annotation_path+"captions_val2017.json",os.path.join(save_dir,noise,str(sev),"coco/annotations/captions_val2017.json"))

                new_instance_annotations = instance_annotations["annotations"].copy()
                perturb = ImagePerturbation(noise, sev=sev).perturb
                print("Starting to generate perturbed images for {} with severity {}".format(noise,sev))
                for i, ann in enumerate(instance_annotations["annotations"]):
                    file_name = id_dict[ann["image_id"]]
                    img = cv2.imread(os.path.join(dataset_val_path,file_name))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = torch.from_numpy(img).float()
                    mask = coco_ann.annToMask(ann)

                    d = img_d_correspondence[file_name]
                    img_perturb, y = perturb(img, mask, d)
                    if(cv2.countNonZero(y)==0):
                        ann_list[i] = 1
                        seg_ann = [0,0,0,0,0,0,0,0]
                        bbox = [0,0,0,0]
                        area = 0.0
                    else:    
                        seg_ann, area, bbox, valid_poly = polygonFromMask(np.array(y))
                        if(valid_poly==0):
                            ann_list[i] = 1
                    # Todo chnage bbox and area
                    new_instance_annotations[i]["bbox"] = bbox
                    new_instance_annotations[i]["area"] = area 

                    new_instance_annotations[i]["segmentation"] = seg_ann
                    # if img is in pil format then convert it to numpy array
                    if isinstance(img_perturb, Image.Image):
                        img_perturb = np.array(img_perturb)
                    img_perturb = cv2.cvtColor(img_perturb, cv2.COLOR_RGB2BGR)
                    save_path = os.path.join(save_dir,noise,str(sev),save_val_dir)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    # Todo make sure that the image is not saved twice
                    if(inst_save_id[file_name] == 0):
                        cv2.imwrite(os.path.join(save_path,file_name),img_perturb)
                        inst_save_id[file_name] = 1
                print(len(ann_list),sum(ann_list))
                new_ann = []
                for i, ann in enumerate(instance_annotations["annotations"]):
                    if(ann_list[i] == 1):
                        continue
                    new_ann.append(new_instance_annotations[i])

                instance_annotations["annotations"] = new_ann
                if(os.path.exists(os.path.join(save_dir,noise,str(sev),"coco/annotations"))==False):
                    os.makedirs(os.path.join(save_dir,noise,str(sev),"coco/annotations"))
                    
                with open(os.path.join(save_dir,noise,str(sev),"coco/annotations","instances_val2017.json"), 'w') as fp:
                    json.dump(instance_annotations, fp)
                
                # For images with no annotations
                img_no_ann = []
                for i in range(len(img_name_list)):
                    if(inst_save_id[img_name_list[i]]==0):
                        img_no_ann.append(img_name_list[i])

                for file_name in img_no_ann:
                    img = cv2.imread(os.path.join(dataset_val_path,file_name))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = torch.from_numpy(img).float()

                    y = np.zeros((img.shape[0],img.shape[1]))
                    d = img_d_correspondence[file_name]
                    img_perturb, _ = perturb(img, mask, d)

                    if isinstance(img_perturb, Image.Image):
                        img_perturb = np.array(img_perturb)
                    img_perturb = cv2.cvtColor(img_perturb, cv2.COLOR_RGB2BGR)
                    save_path = os.path.join(save_dir,noise,str(sev),save_val_dir)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path,file_name),img_perturb)                

                # For panoptic segmentation annotations
                if(noise in ["translate","rotate","shear","dropout"]):
                    perturb = panoptic_perturbation(noise, sev=sev).perturb
                    new_panoptic_ann = panoptic_ann["annotations"].copy()
                    for i, ann in enumerate(panoptic_ann["annotations"]):
                        file_name = id_dict[ann["image_id"]]
                        img = cv2.imread(os.path.join(dataset_path,"panoptic_val2017",file_name.split(".")[0]+".png"))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        valid_seg_list = [0]*len(ann["segments_info"])
                        for j, seg_info in enumerate(ann["segments_info"]):
                            bbox = seg_info["bbox"]
                            d = img_d_correspondence[file_name]
                            mask_aug, bbox_aug, valid_bb = perturb(img, bbox, d)
                            if(valid_bb==0):
                                valid_seg_list[j] = 1
                            new_panoptic_ann[i]["segments_info"][j]["bbox"] = bbox_aug
                            # if img is in pil format then convert it to numpy array
                            if isinstance(mask_aug, Image.Image):
                                mask_aug = np.array(mask_aug)
                            mask_aug = cv2.cvtColor(mask_aug, cv2.COLOR_RGB2BGR)
                            save_path = os.path.join(save_dir,noise,str(sev),"coco/panoptic_val2017")
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            if(panop_save_id[file_name] == 0):
                                cv2.imwrite(os.path.join(save_path,file_name.split(".")[0]+".png"),mask_aug)
                                panop_save_id[file_name] = 1
                        
                        new_seg_info = []
                        for j, seg_info in enumerate(ann["segments_info"]):
                            if(valid_seg_list[j] == 1):
                                continue
                            new_seg_info.append(new_panoptic_ann[i]["segments_info"][j])
                        new_panoptic_ann[i]["segments_info"] = new_seg_info

                    panoptic_ann["annotations"] = new_panoptic_ann
                    if(os.path.exists(os.path.join(save_dir,noise,str(sev),"coco/annotations"))==False):
                        os.makedirs(os.path.join(save_dir,noise,str(sev),"coco/annotations"))
                    with open(os.path.join(save_dir,noise,str(sev),"coco/annotations","panoptic_val2017.json"), 'w') as fp:
                        json.dump(panoptic_ann, fp)
                        

            print("Finished generating perturbed images for {} with severity {}".format(noise,sev))


multiprocessing = True


if multiprocessing:
    l = len(noise_types)//4
    p1 = Process(target=generate_perturbed_images, args=(noise_types[:l],severity))
    p2 = Process(target=generate_perturbed_images, args=(noise_types[l:2*l],severity))
    p3 = Process(target=generate_perturbed_images, args=(noise_types[2*l:3*l],severity))
    p4 = Process(target=generate_perturbed_images, args=(noise_types[3*l:],severity))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
else:
    generate_perturbed_images(noise_types,severity)

print("Finished generating perturbed images")