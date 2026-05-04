import numpy as np
from PIL import Image
from utils.mask_utils import get_class_colors, colorize_mask, blend_masks

def draw_semantic_mask(image, prediction_dict, classes, opacity=0.5):
    """
    Overlay segmentation map, apply colors, and return the annotated image and colors.
    """
    if image is None:
        raise ValueError("Provided image is None.")
        
    if not prediction_dict or 'mask' not in prediction_dict or prediction_dict['mask'] is None:
        raise ValueError("Invalid prediction dictionary or mask is None.")
        
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert('RGB'))
    else:
        image_np = image.copy()
        
    mask = prediction_dict['mask']
    
    # Generate colors with a safety buffer based on num expected classes
    colors = get_class_colors(len(classes) + 10) 
    
    # Convert mask to color overlay
    colored_mask = colorize_mask(mask, colors)
    
    # Alpha blending
    blended_image = blend_masks(image_np, colored_mask, alpha=opacity)
    
    return Image.fromarray(blended_image), colors
