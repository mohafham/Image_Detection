import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_gradcam(model, image_tensor, raw_image_np, target_class):
    """
    Generate Grad-CAM heatmap for AI-generated image detection
    
    Args:
        model: Loaded EfficientNet model
        image_tensor: Preprocessed tensor (1, 3, 224, 224)
        raw_image_np: Original image as float32 numpy array (224,224,3) in [0,1]
        target_class: 0 (Real) or 1 (Fake)
    
    Returns:
        heatmap: Heatmap as numpy array (224,224,3)
    """
    target_layers = [model.blocks[-1]]  # Last conv block of EfficientNet
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
    heatmap = show_cam_on_image(raw_image_np, grayscale_cam, use_rgb=True)
    return heatmap


def generate_gradcam_with_metadata(model, image_tensor, raw_image_np, target_class):
    """
    Generate Grad-CAM heatmap with additional metadata
    
    Args:
        model: Loaded EfficientNet model
        image_tensor: Preprocessed tensor (1, 3, 224, 224)
        raw_image_np: Original image as float32 numpy array (224,224,3) in [0,1]
        target_class: 0 (Real) or 1 (Fake)
    
    Returns:
        dict: Dictionary containing heatmap, grayscale_cam, and top regions
    """
    target_layers = [model.blocks[-1]]  # Last conv block of EfficientNet
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
    heatmap = show_cam_on_image(raw_image_np, grayscale_cam, use_rgb=True)
    
    # Identify top regions of attention
    threshold = np.percentile(grayscale_cam, 90)  # Top 10% attention
    high_attention_mask = grayscale_cam > threshold
    
    # Get regions description
    attention_ratio = (high_attention_mask.sum() / grayscale_cam.size) * 100
    
    return {
        "heatmap": heatmap,
        "grayscale_cam": grayscale_cam,
        "high_attention_mask": high_attention_mask,
        "attention_ratio": attention_ratio,
        "top_regions": f"{attention_ratio:.1f}% of image shows high attention"
    }


def overlay_heatmap(original_image, heatmap, alpha=0.5):
    """
    Overlay heatmap on original image with transparency
    
    Args:
        original_image (np.array): Original image (H, W, 3)
        heatmap (np.array): Heatmap (H, W, 3)
        alpha (float): Transparency factor for heatmap (0-1)
    
    Returns:
        np.array: Blended image
    """
    # Ensure same size
    if original_image.shape != heatmap.shape:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Blend images
    blended = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
    return blended


def create_side_by_side_visualization(original_image, heatmap):
    """
    Create side-by-side visualization of original and heatmap
    
    Args:
        original_image (np.array): Original image
        heatmap (np.array): Grad-CAM heatmap
    
    Returns:
        np.array: Side-by-side visualization
    """
    # Ensure same height
    if original_image.shape[0] != heatmap.shape[0]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Concatenate horizontally
    combined = np.hstack([original_image, heatmap])
    return combined


if __name__ == "__main__":
    print("Grad-CAM utility module for AI-generated image detection")
    print("Usage: Import this module in your inference pipeline")
    print("\nExample:")
    print("  from gradcam import generate_gradcam")
    print("  heatmap = generate_gradcam(model, tensor, img_np, predicted_class)")
