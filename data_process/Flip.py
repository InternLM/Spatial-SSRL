# Processing data for the task Shuffled Patch Reordering

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import IPython
from IPython.display import Image as display_img
import os

def draw_text_with_outline(draw, position, text, font, fill, outline_fill='black', outline_width=2):
    x, y = position
    for dx in [-outline_width, 0, outline_width]:
        for dy in [-outline_width, 0, outline_width]:
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline_fill)
    draw.text(position, text, font=font, fill=fill)

def split_and_flip_one_patch(image_path, border_color=(255, 0, 0), border_thickness=2, flip_index=1):
    num = 2
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    patch_h, patch_w = h // num, w // num

    patches = []
    for i in range(num):
        for j in range(num):
            patch = image[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w].copy()
            patches.append(patch)

    patches[flip_index] = cv2.flip(patches[flip_index], direction)
    
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()

    numbered_patches = []
    for idx, patch in enumerate(patches):
        patch = cv2.rectangle(patch, (0, 0), (patch_w-1, patch_h-1), border_color, thickness=border_thickness)
        pil_patch = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_patch)
        draw_text_with_outline(draw, (10, 10), str(idx), font=font, fill='white')
        patch = cv2.cvtColor(np.array(pil_patch), cv2.COLOR_RGB2BGR)
        numbered_patches.append(patch)

    rows = []
    for i in range(num):
        row = np.hstack(numbered_patches[i*num:(i+1)*num])
        rows.append(row)
    final_image = np.vstack(rows)

    return final_image

# direction decides the flip direction (vertical ofr horizontal)
new_img = split_and_flip_one_patch('source_img/RGB_img.png', flip_index=random.randint(0, 3), direction=0)
cv2.imwrite('Flip.jpg', new_img)