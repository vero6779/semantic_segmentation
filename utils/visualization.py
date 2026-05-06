import numpy as np
from PIL import Image
from utils.mask_utils import get_class_colors, colorize_mask, blend_masks

def draw_semantic_mask(image, prediction_dict, classes, opacity=0.5):
    """
    Overlay segmentation mask on image safely.
    """

    if image is None:
        raise ValueError("Input image is None")

    if prediction_dict is None or not isinstance(prediction_dict, dict):
        raise ValueError("Invalid prediction_dict")

    mask = prediction_dict.get("mask", None)

    if mask is None:
        raise ValueError("Mask is None in draw_semantic_mask")

    if not isinstance(mask, np.ndarray):
        raise ValueError("Mask must be numpy array")

    # Convert image to numpy
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image.copy()

    if not isinstance(image_np, np.ndarray):
        raise ValueError("Image conversion failed")

    # Generate colors
    colors = get_class_colors(len(classes) + 10)

    # Create colored mask
    colored_mask = colorize_mask(mask, colors)

    # Blend
    blended_image = blend_masks(image_np, colored_mask, alpha=opacity)

    return Image.fromarray(blended_image), colors
