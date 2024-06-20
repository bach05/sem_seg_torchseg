import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

import torchvision
from PIL import Image
import colorsys


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_box(box, ax, text=None):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='black', facecolor=(0, 0, 0, 0), lw=2))
    if text:
        ax.text(x0, y0, text, fontsize=10, color='black')



def plot_tensors(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def save_tensorboard_images(images, label, logger, iters, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    logger.add_image(label, grid, iters)

def plot_grid_images(images, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    plt.imshow(grid.detach().permute(1, 2, 0).cpu().numpy())

def get_images_with_mask(images, masks_logits, colors=None, device="cpu", alpha=0.5, **kwargs):

    # print("mask logits: ", masks_logits.shape)

    B, C, H, W = masks_logits.shape

    images = normalizeRGB(images.detach(), use_int8=True).to(device)
    #normalized_masks = torch.nn.functional.softmax(masks_logits.detach(), dim=1)
    mask_idx = torch.argmax(masks_logits.detach(), dim=1).unsqueeze(1)
    # print("mask_idx", mask_idx.shape)
    masks = torch.zeros_like(masks_logits).to(torch.bool).to(device)
    for i in range(C):
        masks[:, i, :, :] = mask_idx[:, 0, :, :] == i

    # print("masks", masks.shape)
    img_with_masks = [
        draw_segmentation_masks(img, masks=mask, colors=colors, alpha=alpha).unsqueeze(0)
        for img, mask in zip(images, masks)
    ]

    return torch.cat(img_with_masks, 0)

def normalizeRGB(images, use_int8=False):

    B, C, H, W = images.shape
    max = torch.max(images.view(B, C, H * W), dim=2)[0]
    min = torch.min(images.view(B, C, H * W), dim=2)[0]
    # print("max: ", max.shape)

    max = max.unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
    min = min.unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
    # print("max: ", max.shape)

    images = (images - min) / (max - min)

    if use_int8:
        images = (images * 255).to(torch.uint8)

    return images

def generate_distinguishable_colors(k):
    colors = ["#000000"] #0 is background
    for i in range(k-1):
        hue = i / k  # Vary the hue component
        saturation = 0.7  # You can adjust this to control saturation
        lightness = 0.6  # You can adjust this to control lightness
        rgb_color = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = "#{:02X}{:02X}{:02X}".format(
            int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255)
        )
        colors.append(hex_color)
    return colors
