# Processing data for the task Cropped Patch Impainting

import cv2
import numpy as np
import random
import os
import json
def interior(crop, side):
    crop_interior_path = "Patch_interior.jpg"
    crop_int = crop[side//4:3*side//4, side//4:3*side//4].copy()
    crop_int = cv2.resize(crop_int, (side, side))
    cv2.imwrite(crop_interior_path,  crop_int)

def large1(img, side, x, y, w, h):
    y_sup = max(0, y - side//4)
    y_up = min(h, y + 5*side//4)
    x_sup = max(0, x - side//4)
    x_up = min(w, x + 5*side//4)
    crop_large1_path = "Patch_exterior.jpg"
    crop_large1 = img[y_sup:y_up, x_sup:x_up].copy()
    crop_large1 = cv2.resize(crop_large1, (side, side))
    cv2.imwrite(crop_large1_path, crop_large1)

def large2(img, side, x, y, w, h):
    y_sup = max(0, y - side//2)
    y_up = min(h, y + 3*side//2)
    x_sup = max(0, x - side//2)
    x_up = min(w, x + 3*side//2)
    crop_large2_path = "Patch_exterior2.jpg"
    crop_large2 = img[y_sup:y_up, x_sup:x_up].copy()
    crop_large2 = cv2.resize(crop_large2, (side, side))
    cv2.imwrite(crop_large2_path,  crop_large2)

def left_rotate(crop, side):
    crop_left_path = "Patch_left_rotate.jpg"
    crop_left = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    crop_left = cv2.resize(crop_left, (side, side))
    cv2.imwrite(crop_left_path,  crop_left)
    
def right_rotate(crop, side):
    crop_right_path = "Patch_right_rotate.jpg"
    crop_right = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    crop_right = cv2.resize(crop_right, (side, side))
    cv2.imwrite(crop_right_path,  crop_right)


def random_crop_blacken(img_path):
    all_processes = ['left_rotate', 'right_rotate', 'interior', 'large1', 'large2']
    correct = [0, 1, 2, 3]
    gt = random.choice(correct)

    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]

    side = int(min(h, w) / 2)

    x = random.randint(0, w - side)
    y = random.randint(0, h - side)

    crop = img[y:y+side, x:x+side].copy()

    img_black = img.copy()
    img_black[y:y+side, x:x+side] = 0

    black_path = "Cropped.jpg"
    cv2.imwrite(black_path, img_black)
    for j in range(4):
        if j==gt:
            crop_path  = "gt_patch.jpg"
            cv2.imwrite(crop_path,  crop)
        else:
            fun = random.choice(all_processes)
            all_processes.remove(fun) 
            if fun=='left_rotate':
                left_rotate(crop, side)
            elif fun=='right_rotate':
                right_rotate(crop, side)
            elif fun=='interior':
                interior(crop, side)
            elif fun=='large1':
                large1(img, side, x, y, w, h)
            else:
                large2(img, side, x, y, w, h)

    return gt
img_path = "source_img/RGB_img.png"
gt = random_crop_blacken(img_path)



