# Processing data for the task Regional Depth Ordering
# Source data from DIODE dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import os

small = 0.5
large = 99.5

def process_sample_3(img_path, depth_path, mask_path):
    depth = np.load(depth_path)
    mask  = np.load(mask_path)
    depth = np.squeeze(depth, axis=-1)

    if mask.shape[1] != depth.shape[1]:
        mask = mask[:, :depth.shape[1]]
    assert mask.shape == depth.shape

    valid = mask.astype(bool)
    valid_depth = depth[valid]

    if valid_depth.size == 0:
        return None
    d_min = np.percentile(valid_depth, small)
    d_max = np.percentile(valid_depth, large)

    depth_norm = np.full_like(depth, np.nan, dtype=np.float32)
    within = valid & (depth >= d_min) & (depth <= d_max)
    depth_norm[within] = (depth[within] - d_min) / (d_max - d_min)

    def check_patch(depth_norm, y, x, patch_size=25, diff=0.15):
        half = patch_size // 2
        H, W = depth_norm.shape
        y0, y1 = y - half, y + half + 1
        x0, x1 = x - half, x + half + 1
        if y0 < 0 or y1 > H or x0 < 0 or x1 > W:
            return False, None
        patch = depth_norm[y0:y1, x0:x1]
        if np.isnan(patch).any():
            return False, None
        if patch.max() - patch.min() >= diff:
            return False, None
        return True, patch

    ranges = [(0.0, 0.3), (0.35, 0.65), (0.7, 1.0)]
    labels = [1, 2, 3]
    found = {}
    for (low, high), label in zip(ranges, labels):
        candidates = np.argwhere((depth_norm >= low) & (depth_norm <= high) & (~np.isnan(depth_norm)))
        np.random.shuffle(candidates)
        ok = False
        for y, x in candidates:
            ok, _ = check_patch(depth_norm, y, x)
            if ok:
                found[label] = (y, x)
                break
        if not ok:
            for y, x in candidates:
                ok, _ = check_patch(depth_norm, y, x, patch_size=11, diff=0.2)
                if ok:
                    found[label] = (y, x)
                    break
            if not ok:
                print(f"No available regions")
                return None

    if len(found)!=3:
        print("No available regions")
        return None
    
    jpg = plt.imread(img_path)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(jpg)
    ax.axis('off')

    lst = [1, 2, 3]
    random.shuffle(lst)

    for i in range(3):
        (y, x) = found[lst[i]]
        ax.text(x, y, str(i+1), color='black', fontsize=15, ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.3", facecolor='white', edgecolor='black', lw=1.5))

    plt.savefig("Depth.jpg", bbox_inches='tight', dpi=150)
    plt.close()
    return np.argsort(lst)+1

img_path = "source_img/RGB-D_img3.png"
depth_path = "source_img/RGB-D_depth3.npy"
mask_path = "source_img/RGB-D_depth_mask3.npy"
gt = process_sample_3(img_path, depth_path, mask_path)
print(gt)