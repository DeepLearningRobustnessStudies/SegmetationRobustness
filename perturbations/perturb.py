import numpy as np
import cv2
import torch
import skimage
import torchvision.datasets as datasets
import random
import math
from torchvision import transforms

from PIL import Image
from io import BytesIO
from scipy.ndimage import zoom as scizoom
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import sys

from PIL import Image, ImageOps, ImageEnhance


import ctypes


class ImagePerturbation(torch.nn.Module):
    def __init__(self,perturb_type,sev=1,train=False):
        super(ImagePerturbation, self).__init__()
        self.perturb_type = perturb_type
        self.sev = sev
        self.train = train
        self.mapping = {"gaussian":self.gaussian_noise,
                        "shot":self.shot_noise,
                        "impulse":self.impulse_noise,
                        "speckle":self.speckle_noise,
                        "defocus":self.defocus_blur,
                        "motion":self.motion_blur,
                        "zoom":self.zoom_blur,
                        "contrast":self.contrast,
                        "brightness":self.brightness,
                        "darkness":self.darkness,
                        "rotate":self.rotate,
                        "translate":self.translate,
                        "shear":self.shear,
                        "fog":self.fog,
                        "snow":self.snow,
                        "jpeg":self.jpeg,
                        "pixelate":self.pixelate,
                        "dropout":self.course_dropout,
                        }
        self.perturb = self.mapping[self.perturb_type]

    def forward(self, x, y):
        x, y = self.perturb(x, y)
        return x, y


    def gaussian_noise(self, x, y):
        c = [.08, .12, 0.18, 0.26, 0.38][self.sev - 1]
        x = np.array(x)/255.
        x = Image.fromarray(np.uint8(np.clip(x + np.random.normal(0, c, x.shape), 0, 1)*255))
        return x, y
    
    def shot_noise(self, x, y):
        c = [250, 100, 50, 30, 15][self.sev - 1]

        x = np.array(x) / 255.
        return Image.fromarray(np.uint8(np.clip(np.random.poisson(x * c) / c, 0, 1) * 255)), y
    
    def speckle_noise(self, x, y):
        c = [.15, .2, 0.25, 0.3, 0.35][self.sev - 1]

        x = np.array(x) / 255.
        return Image.fromarray(np.uint8(np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255)), y
    
    def impulse_noise(self, x, y):
        c = [.03, .06, .09, 0.17, 0.27][self.sev - 1]

        x = skimage.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255)), y

    def jpeg(self, x, y):
        c = [25, 18, 15, 10, 7][self.sev - 1]
        x = np.array(x).astype(np.uint8)
        x = Image.fromarray(x)

        output = BytesIO()
        x.save(output, 'JPEG', quality=c)
        x = Image.open(output), y

        return x

    def pixelate(self, x, y):
        c = [1, 2, 3, 4, 5][self.sev - 1]
        x = np.array(x).astype(np.uint8)
 
        aug = iaa.imgcorruptlike.Pixelate(severity=c)
        x = aug.augment_image(x)
        return Image.fromarray(x), y
    
    def darkness(self, x, y):
        c = [.9, .8, .7, .6, .5][self.sev - 1]

        x = Image.fromarray((np.array(x)*1).astype(np.uint8))
        return ImageEnhance.Brightness(x).enhance(c), y

    def brightness(self, x, y):
        c = [1, 2, 3, 4, 5][self.sev - 1]

        x = np.array(x).astype(np.uint8)
        aug = iaa.imgcorruptlike.Brightness(severity=c)
        x = aug.augment_image(x)
        return Image.fromarray(x), y


    def contrast(self, x, y):
        c = [1.2, 1.4, 1.6, 1.8, 2.0][self.sev - 1]

        x = Image.fromarray((np.array(x)*1).astype(np.uint8))
        x = ImageEnhance.Contrast(x).enhance(c)
        return x, y

    def fog(self, x, y):
        c = [1, 2, 3, 4, 5][self.sev - 1]
        x = np.array(x).astype(np.uint8)
        aug = iaa.imgcorruptlike.Fog(severity=c)
        x = aug.augment_image(x)

        return Image.fromarray(x), y

    def snow(self, x, y):
        c = [1, 2, 3, 4, 5][self.sev - 1]
        # seq = iaa.Sequential([
        #     iaa.Snowflakes(flake_size=c,speed=(0.007, 0.03)),
        # ])
        seq = iaa.imgcorruptlike.Snow(severity=c)
        x = np.array(x).astype(np.uint8)
        # x, y = seq(image=x, segmentation_maps=y)
        x = seq.augment_image(x)
        return Image.fromarray(x), y
    
    def rotate(self, x, y, d, ade=False):
        c = [20, 30, 40, 50, 60][self.sev - 1]
        dx = d[0]

        x = np.array(x)
        seq = iaa.Sequential([
            iaa.Affine(rotate=c*dx),
        ])
        
        if(not ade):
            x = seq(image=x)
            if(y is not None):
                y = seq(image=y)
        else:
            y = np.array(y).astype(np.int32)
            y = SegmentationMapsOnImage(y, shape=y.shape)
            x, y = seq(image=x, segmentation_maps=y)
            y = y.get_arr().astype(np.uint8)
        return Image.fromarray((x*1).astype(np.uint8)), y

    def translate(self, x, y, d, ade=False):
        c = [20, 40, 60, 80, 100][self.sev - 1]
        dx, dy = d[0], d[1]
        seq = iaa.Sequential([
            iaa.Affine(translate_px={"x": c*dx, "y": c*dy}),
        ])
        x = np.array(x) 
        if(not ade):
            x = seq(image=x)
            if(y is not None):
                y = seq(image=y)
        else:
            y = np.array(y).astype(np.int32)
            y = SegmentationMapsOnImage(y, shape=y.shape)
            x, y = seq(image=x, segmentation_maps=y)
            y = y.get_arr().astype(np.uint8)

        return Image.fromarray((x*1).astype(np.uint8)), y

    def shear(self, x, y, d, ade=False):
        c = [5, 10, 15, 20, 25][self.sev - 1]
        dx = d[0]
        seq = iaa.Sequential([
            iaa.Affine(shear=c*dx),
        ])
        x = np.array(x)
        if(not ade):
            x = seq(image=x)
            if(y is not None):
                y = seq(image=y)
        else:
            y = np.array(y).astype(np.int32)
            y = SegmentationMapsOnImage(y, shape=y.shape)
            x, y = seq(image=x, segmentation_maps=y)
            y = y.get_arr().astype(np.uint8)
        return Image.fromarray((x*1).astype(np.uint8)), y
    
    def motion_blur(self, x, y):
        c = [7, 9, 13, 17, 25][self.sev - 1]
        seq = iaa.Sequential([
            iaa.MotionBlur(k=c),
        ])
        x = np.array(x)
        # y = np.max(y, axis=3).squeeze(0).astype(np.uint8)
        x = seq(image = x)
        if(y != None):
            y = seq(image = y)
        return Image.fromarray((x*1).astype(np.uint8)), y


    def disk(self, radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)
    
    def defocus_blur(self,x, y):
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][self.sev - 1]

        x = np.array(x) / 255.0
        kernel = self.disk(radius=c[0], alias_blur=c[1])
        # *255
        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3
        return Image.fromarray((np.clip(channels, 0, 1) * 255).astype(np.uint8)), y
    
    def clipped_zoom(self, img, zoom_factor):

        ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
        top0 = (img.shape[0] - ch0) // 2

        # clipping along the height dimension:
        ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
        top1 = (img.shape[1] - ch1) // 2

        img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
                    (zoom_factor, zoom_factor, 1), order=1)

        return img

    def zoom_blur(self, x, y):
        c = [np.arange(1, 1.05, 0.01),
             np.arange(1, 1.07, 0.01),
             np.arange(1, 1.10, 0.02),
             np.arange(1, 1.12, 0.02),
             np.arange(1.03, 1.18, 0.03)][self.sev - 1]
    
        x = (np.array(x) / 255.0).astype(np.float32)
        out = np.zeros_like(x)

        for zoom_factor in c:
            zoom_layer = self.clipped_zoom(x, zoom_factor)
            zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], :]
            try:
                out += zoom_layer
            except ValueError:
                out[:zoom_layer.shape[0], :zoom_layer.shape[1]] += zoom_layer
        
        # for zoom_factor in c:
        #     out += self.clipped_zoom(x, zoom_factor)

        x = (x + out) / (len(c) + 1)
        
        return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255.0)), y

    def course_dropout(self, x, y):
        c = [0.02, 0.04, 0.06, 0.08, 0.1][self.sev - 1]
        seq = iaa.Sequential([
            iaa.CoarseDropout(c,size_percent=(0.2, 0.25)),
        ])
        x = np.array(x)
        y = y = np.max(y, axis=3).squeeze(0).astype(np.uint8)
        x  = seq(image=x)
        y = seq(image=y)
        return Image.fromarray((x*1).astype(np.uint8)), y


class panoptic_perturbation(torch.nn.Module):
    def __init__(self,perturb_type,sev=1,train=False):
        super(panoptic_perturbation, self).__init__()
        self.perturb_type = perturb_type
        self.sev = sev
        self.train = train
        self.mapping = {"rotate":self.rotate,
                        "translate":self.translate,
                        "shear":self.shear,
                        # "dropout":self.course_dropout,
                        }
        self.perturb = self.mapping[self.perturb_type]

    
    def rotate(self, mask, bbox, d):

        xl, yl, xr, yr = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
        bb = BoundingBox(xl, yl, xr, yr)
        bb = BoundingBoxesOnImage([bb], shape=mask.shape)

        c = [20, 30, 40, 50, 60][self.sev - 1]
        dx = d[0]

        mask = np.array(mask)
        seq = iaa.Sequential([
            iaa.Affine(rotate=c*dx),
        ])
        valid_bb = True
        mask_aug, bb_aug = seq(image=mask, bounding_boxes=bb)
        bb_aug = bb_aug.remove_out_of_image(True, False).clip_out_of_image()
        if(len(bb_aug.bounding_boxes)>0):
            bbx = bb_aug.bounding_boxes[0]
            bbox = [bbx.x1, bbx.y1, bbx.x2-bbx.x1, bbx.y2-bbx.y1]
        elif(len(bb_aug.bounding_boxes)==0):
            bbox = [0, 0, 0, 0]
            valid_bb = False
        mask_aug = Image.fromarray((mask_aug*1).astype(np.uint8))
        return mask_aug, bbox, valid_bb


    def translate(self, mask, bbox, d):
        xl, yl, xr, yr = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
        bb = BoundingBox(xl, yl, xr, yr)
        bbs = BoundingBoxesOnImage([bb], shape=mask.shape)

        c = [20, 40, 60, 80, 100][self.sev - 1]
        dx, dy = d[0], d[1]

        mask = np.array(mask)
        seq = iaa.Sequential([
            iaa.Affine(translate_px={"x": c*dx, "y": c*dy}),
        ])

        valid_bb = True
        mask_aug, bb_aug = seq(image=mask, bounding_boxes=bbs)
        bb_aug = bb_aug.remove_out_of_image(True, False).clip_out_of_image()
        if(len(bb_aug.bounding_boxes)>0):
            bbx = bb_aug.bounding_boxes[0]
            bbox = [bbx.x1, bbx.y1, bbx.x2-bbx.x1, bbx.y2-bbx.y1]
        elif(len(bb_aug.bounding_boxes)==0):
            bbox = [0, 0, 0, 0]
            valid_bb = False
        mask_aug = Image.fromarray((mask_aug*1).astype(np.uint8))
        return mask_aug, bbox, valid_bb
        
    def shear(self, mask, bbox, d):
        xl, yl, xr, yr = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
        bb = BoundingBox(xl, yl, xr, yr)
        bb = BoundingBoxesOnImage([bb], shape=mask.shape)

        c = [5, 10, 15, 20, 25][self.sev - 1]
        dx = d[0]

        mask = np.array(mask)
        seq = iaa.Sequential([
            iaa.Affine(shear=c*dx),
        ])

        valid_bb = True
        mask_aug, bb_aug = seq(image=mask, bounding_boxes=bb)
        bb_aug = bb_aug.remove_out_of_image(True, False).clip_out_of_image()
        if(len(bb_aug.bounding_boxes)>0):
            bbx = bb_aug.bounding_boxes[0]
            bbox = [bbx.x1, bbx.y1, bbx.x2-bbx.x1, bbx.y2-bbx.y1]
        elif(len(bb_aug.bounding_boxes)==0):
            bbox = [0, 0, 0, 0]
            valid_bb = False
        mask_aug = Image.fromarray((mask_aug*1).astype(np.uint8))
        return mask_aug, bbox, valid_bb

    
    






