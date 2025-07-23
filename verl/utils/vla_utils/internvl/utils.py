from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import torchvision.transforms.functional as TVF
from typing import Tuple
from PIL import Image
import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INTERNVL_IMG_SIZE = 448
INTERNVL_IMG_MAX_NUM = 12
INTERNVL_IMG_USE_THUMBNAIL = True


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

INTERNVL_IMG_TRANSFORM = build_transform(INTERNVL_IMG_SIZE)

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def pad_to_square(image: Image.Image, resize_target: Tuple[int, int] = (480, 480), padding_fill_value: Tuple[int, int, int] = (123, 116, 103)) -> Image.Image:
    """
    Pad a PIL.Image to square shape and resize, using a symmetric border and a fill value.
    Args:
        image: PIL.Image.Image, input image
        resize_target: tuple, target size after resize
        padding_fill_value: tuple, RGB fill value for padding
    Returns:
        PIL.Image.Image, padded and resized image
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    w, h = image.size
    max_wh = max(w, h)
    horizontal_pad = (max_wh - w) // 2
    vertical_pad = (max_wh - h) // 2
    padding = (horizontal_pad, vertical_pad, max_wh - w - horizontal_pad, max_wh - h - vertical_pad)
    padded_image = TVF.pad(image, padding, fill=padding_fill_value, padding_mode="constant")
    resized_image = padded_image.resize(resize_target, Image.BILINEAR)
    return resized_image


def process_image_internvl(image: np.ndarray):
    image = Image.fromarray(image).convert("RGB")
    image = pad_to_square(image, resize_target=(INTERNVL_IMG_SIZE, INTERNVL_IMG_SIZE))
    images = dynamic_preprocess(image, image_size=INTERNVL_IMG_SIZE, use_thumbnail=INTERNVL_IMG_USE_THUMBNAIL, max_num=INTERNVL_IMG_MAX_NUM)
    pixel_values = [INTERNVL_IMG_TRANSFORM(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values