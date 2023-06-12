import os
import shutil
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from perturb import *
import json
from PIL import Image
from pathlib import Path
from multiprocessing import Pool, Process, Manager, Value
import argparse

import sys
# sys.path.append("../models/ODISE/datasets")
from models.ODISE.datasets.prepare_ade20k_sem_seg import *
from models.ODISE.datasets.prepare_ade20k_ins_seg import *
from models.ODISE.datasets.prepare_ade20k_pan_seg import *


def get_parser():
    parser = argparse.ArgumentParser(description="Perturbations")
    parser.add_argument("--ade_dataset_path", type=str, default="../datasets/ADE20K_2016/ade/ADEChallengeData2016/", help="Path to ADE20K dataset")
    parser.add_argument("--save_dir", type=str, default="/home/c3-0/datasets/ADE20K_150_perturbated/", help="Path to save perturbed images")

    return parser

noise_types = ["gaussian","shot","impulse","speckle","defocus","motion","zoom","contrast","brightness","darkness","jpeg","pixelate","fog","snow","rotate","translate","shear"]
severity = [1,2,3,4,5]

args = get_parser().parse_args()
dataset_path = args.ade_dataset_path
save_dir = args.save_dir


img_d_correspondence = {}
for file_name in os.listdir(dataset_val_path):
    img_d_correspondence[file_name[:-4]] = np.random.choice([-1,1],2)


def generate_perturbed_images(noise_type, severity):
    for noise in noise_type:
        for sev in severity:
            if(noise not in ["translate","rotate","shear","dropout"]):
                save_perturb_path = os.path.join(save_dir, noise, str(sev),"ade/ADEChallengeData2016/")
                if(os.path.exists(os.path.join(save_dir, noise, str(sev)))):
                    shutil.rmtree(os.path.join(save_dir, noise, str(sev)))
                os.makedirs(save_perturb_path)

                shutil.copytree(dataset_path + "annotations/validation/", save_perturb_path + "annotations/validation/")
                shutil.copytree(dataset_path + "annotations_instance/validation/", save_perturb_path + "annotations_instance/validation/")
                shutil.copytree(dataset_path + "annotations_detectron2/validation/", save_perturb_path + "annotations_detectron2/validation/")
                shutil.copytree(dataset_path + "ade20k_panoptic_val/", save_perturb_path + "ade20k_panoptic_val/")

                shutil.copy(dataset_path + "ade20k_panoptic_val.json", save_perturb_path + "ade20k_panoptic_val.json")
                shutil.copy(dataset_path + "ade20k_instance_val.json", save_perturb_path + "ade20k_instance_val.json")
                shutil.copy(dataset_path + "objectInfo150.txt", save_perturb_path + "objectInfo150.txt")
                shutil.copy(dataset_path + "sceneCategories.txt", save_perturb_path + "sceneCategories.txt")
                
                perturb = ImagePerturbation(noise, sev=sev).perturb
                print("Starting to generate perturbed images for {} with severity {}".format(noise,sev))

                for i, file_name in enumerate(os.listdir(dataset_val_path)):
                    img = cv2.imread(dataset_val_path + file_name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = torch.from_numpy(img).float()
                    # perturb = ImagePerturbation(noise, sev=sev)
                    img_perturb, _ = perturb(img, None)
                    if isinstance(img_perturb, Image.Image):
                        img_perturb = np.array(img_perturb)
                    img_perturb = cv2.cvtColor(img_perturb, cv2.COLOR_RGB2BGR)
                    save_path = os.path.join(save_perturb_path,"images/validation")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path,file_name),img_perturb)

                print("Finished generating perturbed images for {} with severity {}".format(noise,sev))

            else:
                save_perturb_path = os.path.join(save_dir, noise, str(sev),"ade/ADEChallengeData2016/")
                if(os.path.exists(os.path.join(save_dir, noise, str(sev)))):
                    shutil.rmtree(os.path.join(save_dir, noise, str(sev)))
                os.makedirs(save_perturb_path)

                shutil.copy(dataset_path + "objectInfo150.txt", save_perturb_path + "objectInfo150.txt")
                shutil.copy(dataset_path + "sceneCategories.txt", save_perturb_path + "sceneCategories.txt")
                
                perturb = ImagePerturbation(noise, sev=sev).perturb
                print("Starting to generate perturbed images for {} with severity {}".format(noise,sev))

                for i, file_name in enumerate(os.listdir(dataset_val_path)):
                    img = cv2.imread(dataset_val_path + file_name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = torch.from_numpy(img).float()
                    d = img_d_correspondence[file_name[:-4]]
                    img_perturb, _ = perturb(img, None, d)
                    if isinstance(img_perturb, Image.Image):
                        img_perturb = np.array(img_perturb)
                    img_perturb = cv2.cvtColor(img_perturb, cv2.COLOR_RGB2BGR)
                    save_path = os.path.join(save_perturb_path,"images/validation")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path,file_name),img_perturb)

                    ann_val_img = cv2.imread(dataset_path + "annotations/validation/" + file_name[:-4] + ".png",0)
                    # ann_val_img = cv2.cvtColor(ann_val_img, cv2.COLOR_BGR2RGB)
                    ann_val_img = torch.from_numpy(ann_val_img).float()
                    _, ann_val_img_perturb = perturb(img, ann_val_img, d, True)
                    if isinstance(ann_val_img_perturb, Image.Image):
                        ann_val_img_perturb = np.array(ann_val_img_perturb)
                    # ann_val_img_perturb = cv2.cvtColor(ann_val_img_perturb, cv2.COLOR_RGB2BGR)
                    save_path = os.path.join(save_perturb_path,"annotations/validation")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path,file_name[:-4] + ".png"),ann_val_img_perturb)

                    ann_inst_val_img = cv2.imread(dataset_path + "annotations_instance/validation/" + file_name[:-4] + ".png")
                    ann_inst_val_img = cv2.cvtColor(ann_inst_val_img, cv2.COLOR_BGR2RGB)
                    ann_inst_val_img = torch.from_numpy(ann_inst_val_img).float()
                    _, ann_inst_val_img_perturb = perturb(img, ann_inst_val_img, d, True)
                    if(isinstance(ann_inst_val_img_perturb, Image.Image)):
                        ann_inst_val_img_perturb = np.array(ann_inst_val_img_perturb)
                    ann_inst_val_img_perturb = cv2.cvtColor(ann_inst_val_img_perturb, cv2.COLOR_RGB2BGR)
                    save_path = os.path.join(save_perturb_path,"annotations_instance/validation")
                    if(not os.path.exists(save_path)):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path,file_name[:-4] + ".png"),ann_inst_val_img_perturb)

                # generate sem_seg, inst_seg, pan_seg annotations
                sem_seg_dataset_dir = (
                    Path(save_dir + noise +"/"+str(sev)) / "ade" / "ADEChallengeData2016"
                )
                gen_ade20k_sem_seg(sem_seg_dataset_dir)

                inst_pan_seg_dataset_dir = os.path.join(save_dir,noise,str(sev))
                gen_ade20k_ins_seg(inst_pan_seg_dataset_dir)
                gen_ade20k_pan_seg(inst_pan_seg_dataset_dir)

                print("Finished generating perturbed images for {} with severity {}".format(noise,sev))


multiprocessing = True


if multiprocessing:
    l = len(noise_types)//4
    p1 = Process(target=generate_perturbed_images, args=(noise_types[0:l],severity))
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
