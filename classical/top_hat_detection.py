"""
Crack Detection using Morphological Top-Hat Transform
Based on: Giakoumis et al., IEEE TIP 2006
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_cracks(image_path, threshold=23, se_size=3, n_dilations=2, output_dir="results"):
    """
    Detect cracks in a digitized painting using the black top-hat transform.

    Args:
        image_path  : path to input painting image
        threshold   : binarization threshold (tune per image, typically 15-30)
        se_size     : structuring element size (default 3x3 as in paper)
        n_dilations : number of dilations to build final SE (default 2)
        output_dir  : folder to save results

    Returns:
        crack_mask  : binary mask where 255 = crack pixel
        luminance   : grayscale luminance channel
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Step 1: Extract luminance (L channel from LAB) ---
    # Paper works on luminance component
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    luminance = img_lab[:, :, 0]  # L channel

    # --- Step 2: Build structuring element via repeated dilation ---
    # nB = B ⊕ B ⊕ ... ⊕ B (n times) as in Equation (2) of the paper
    base_se = cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, se_size))
    final_se = base_se.copy()
    for _ in range(n_dilations - 1):
        final_se = cv2.dilate(final_se, base_se)

    # --- Step 3: Black top-hat = closing(f) - f ---
    # Highlights dark structures (cracks) against lighter background
    # Equivalent to applying white top-hat on negated image
    closed = cv2.morphologyEx(luminance, cv2.MORPH_CLOSE, final_se)
    top_hat = cv2.subtract(closed, luminance)  # Equation (3) from paper

    # --- Step 4: Threshold to get binary crack mask ---
    _, crack_mask = cv2.threshold(top_hat, threshold, 255, cv2.THRESH_BINARY)
    crack_mask = crack_mask.astype(np.uint8)

    # --- Step 5: Size-based noise removal (from arXiv 2602.12742) ---
    # Remove connected components smaller than 5 pixels (noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crack_mask, connectivity=8)
    clean_mask = np.zeros_like(crack_mask)
    for label in range(1, num_labels):  # skip background (label 0)
        if stats[label, cv2.CC_STAT_AREA] >= 5:
            clean_mask[labels == label] = 255

    # --- Save outputs ---
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(os.path.join(output_dir, f"{base_name}_tophat.png"), top_hat)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_crack_mask.png"), clean_mask)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Painting")
    axes[0].axis("off")

    axes[1].imshow(top_hat, cmap="gray")
    axes[1].set_title(f"Top-Hat Transform Output")
    axes[1].axis("off")

    axes[2].imshow(clean_mask, cmap="gray")
    axes[2].set_title(f"Binary Crack Mask (T={threshold})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_detection_result.png"), dpi=150)
    plt.close()

    print(f"[Detection] Crack pixels detected: {np.sum(clean_mask > 0)}")
    print(f"[Detection] Results saved to: {output_dir}/")

    return clean_mask, luminance


if __name__ == "__main__":
    import sys
    # Usage: python top_hat_detection.py <image_path> [threshold]
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_painting.jpg"
    threshold  = int(sys.argv[2]) if len(sys.argv) > 2 else 23

    mask, lum = detect_cracks(image_path, threshold=threshold)
    print("Detection complete.")
