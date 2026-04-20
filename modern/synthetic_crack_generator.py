"""
Synthetic Craquelure Generator
Based on: Cuch-Guillen et al., arXiv:2602.12742, Feb 2026

Generates realistic synthetic crack masks over clean paintings using:
- Cubic Bezier trajectories with Gaussian-perturbed control points
- Tapered geometry (thicker mid-section, thinner ends)
- Branching patterns
- Morphological refinement + Gaussian blurring for realism

This module solves the annotation scarcity problem:
no labeled crack data needed — we generate our own training pairs.
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple, List


def cubic_bezier(p0, p1, p2, p3, n_points=120):
    """
    Sample points along a cubic Bezier curve.
    Equation (6) from arXiv 2602.12742:
    B(t) = (1-t)^3 * p0 + 3(1-t)^2*t * p1 + 3(1-t)*t^2 * p2 + t^3 * p3

    Args:
        p0, p3 : endpoints
        p1, p2 : control points (perturbed by Gaussian noise)
        n_points: number of sample points along the curve

    Returns:
        points: array of (x, y) coordinates along the curve
    """
    t = np.linspace(0, 1, n_points)
    points = (
        (1 - t)**3 * p0[:, None] +
        3 * (1 - t)**2 * t * p1[:, None] +
        3 * (1 - t) * t**2 * p2[:, None] +
        t**3 * p3[:, None]
    ).T  # shape: (n_points, 2)
    return points.astype(np.int32)


def draw_tapered_crack(mask, points, alpha=2.0, sigma_r=0.5):
    """
    Draw a single crack with tapered geometry.
    Equation (7) from arXiv 2602.12742:
    r(t) ~ N(alpha * (1 - |t - 0.5|), sigma_r^2)
    
    Thicker in the middle, thinner at the ends — mimics real craquelure.

    Args:
        mask   : binary mask to draw on (H x W)
        points : Bezier curve sample points
        alpha  : max radius scale (default 2.0 px)
        sigma_r: radius noise std (default 0.5 px)
    """
    n = len(points)
    for i, (x, y) in enumerate(points):
        t = i / max(n - 1, 1)
        # Tapered radius: thicker at center, thinner at tips
        mean_r = alpha * (1 - abs(t - 0.5))
        r = max(1, int(np.random.normal(mean_r, sigma_r)))
        # Bounds check
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            cv2.circle(mask, (int(x), int(y)), r, 255, -1)


def generate_branch(start_point, direction, img_shape, length_scale=0.3):
    """
    Spawn a branch crack from a point on an existing crack.
    Branch is shorter and at an angle to the parent direction.

    Args:
        start_point  : (x, y) origin of the branch
        direction    : direction vector of parent crack at branch point
        img_shape    : (H, W) of the image
        length_scale : fraction of image diagonal for branch length

    Returns:
        Bezier control points for the branch
    """
    H, W = img_shape[:2]
    diag = np.sqrt(H**2 + W**2)

    # Rotate direction by random angle ±30–60 degrees
    angle = np.random.uniform(30, 60) * np.random.choice([-1, 1])
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_dir = np.array([
        cos_a * direction[0] - sin_a * direction[1],
        sin_a * direction[0] + cos_a * direction[1]
    ])
    rot_dir = rot_dir / (np.linalg.norm(rot_dir) + 1e-8)

    branch_len = diag * length_scale * np.random.uniform(0.3, 0.7)
    p0 = np.array(start_point, dtype=float)
    p3 = p0 + rot_dir * branch_len

    # Control points with Gaussian perturbation (sigma=8px as in paper)
    sigma_p = 8.0
    p1 = p0 + rot_dir * branch_len * 0.33 + np.random.normal(0, sigma_p, 2)
    p2 = p0 + rot_dir * branch_len * 0.66 + np.random.normal(0, sigma_p, 2)

    # Clip to image bounds
    p3 = np.clip(p3, [0, 0], [W - 1, H - 1])

    return p0, p1, p2, p3


def generate_single_crack(img_shape, p_branch=0.4):
    """
    Generate one crack trajectory (+ optional branches).

    Args:
        img_shape : (H, W) of target image
        p_branch  : probability of spawning a branch [0.3, 0.5] per paper

    Returns:
        List of (p0, p1, p2, p3) tuples for each Bezier segment
    """
    H, W = img_shape[:2]

    # Random endpoints anywhere in the image
    p0 = np.array([np.random.uniform(0, W), np.random.uniform(0, H)])
    p3 = np.array([np.random.uniform(0, W), np.random.uniform(0, H)])

    # Control points perturbed by Gaussian noise (sigma=8px as in paper)
    sigma_p = 8.0
    p1 = p0 + np.random.normal(0, sigma_p, 2)
    p2 = p3 + np.random.normal(0, sigma_p, 2)

    # n_points: 80 to 180 depending on crack length
    crack_len = np.linalg.norm(p3 - p0)
    n_points = int(np.clip(crack_len / 3, 80, 180))

    segments = [(p0, p1, p2, p3, n_points)]

    # Optionally spawn a branch
    if np.random.random() < p_branch:
        # Pick a random point on the curve as branch origin
        t_branch = np.random.uniform(0.3, 0.7)
        idx = int(t_branch * (n_points - 1))
        points = cubic_bezier(p0, p1, p2, p3, n_points)
        if idx < len(points):
            branch_origin = points[idx]
            # Approximate local direction
            if idx + 1 < len(points):
                direction = (points[idx + 1] - points[idx]).astype(float)
            else:
                direction = (points[idx] - points[idx - 1]).astype(float)
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            bp0, bp1, bp2, bp3 = generate_branch(branch_origin, direction, img_shape)
            b_len = np.linalg.norm(bp3 - bp0)
            b_points = int(np.clip(b_len / 3, 40, 100))
            segments.append((bp0, bp1, bp2, bp3, b_points))

    return segments


def generate_crack_mask(img_shape, n_cracks=None, p_branch=0.4, alpha=2.0, sigma_r=0.5):
    """
    Generate a complete synthetic crack mask for one image.

    Args:
        img_shape : (H, W) or (H, W, C)
        n_cracks  : number of crack trajectories (default: random 80-150 per paper)
        p_branch  : branch probability per crack
        alpha     : tapered radius scale
        sigma_r   : radius noise

    Returns:
        crack_mask : binary uint8 mask (H x W), 255 = crack
        raw_mask   : raw (before refinement) for comparison
    """
    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    if n_cracks is None:
        n_cracks = np.random.randint(80, 151)  # 80-150 as per paper

    for _ in range(n_cracks):
        segments = generate_single_crack((H, W), p_branch=p_branch)
        for seg in segments:
            p0, p1, p2, p3, n_pts = seg
            points = cubic_bezier(p0, p1, p2, p3, n_pts)
            draw_tapered_crack(mask, points, alpha=alpha, sigma_r=sigma_r)

    raw_mask = mask.copy()

    # --- Refinement Step 1: Morphological erosion (2x2) to refine thickness ---
    erode_kernel = np.ones((2, 2), np.uint8)
    mask = cv2.erode(mask, erode_kernel, iterations=1)

    # --- Refinement Step 2: Gaussian blur (5x5, sigma=2) to soften edges ---
    mask_float = mask.astype(np.float32)
    mask_blurred = cv2.GaussianBlur(mask_float, (5, 5), 2)

    # --- Final binarization at threshold 50 ---
    _, final_mask = cv2.threshold(mask_blurred, 50, 255, cv2.THRESH_BINARY)
    final_mask = final_mask.astype(np.uint8)

    return final_mask, raw_mask


def apply_crack_to_image(image, crack_mask, crack_gray_value=None):
    """
    Apply synthetic crack mask to a clean painting to create a damaged image.
    Replaces crack pixels with a dark gray value simulating paint loss.

    Args:
        image           : clean BGR painting image
        crack_mask      : binary mask (255 = crack)
        crack_gray_value: gray value for crack pixels (auto if None)

    Returns:
        damaged_image : painting with synthetic cracks applied
    """
    damaged = image.copy()

    if crack_gray_value is None:
        # Estimate from image: cracks are typically darker than surroundings
        # Use ~20% of mean luminance as crack color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        crack_gray_value = max(10, int(np.mean(gray) * 0.20))

    crack_color = np.array([crack_gray_value, crack_gray_value, crack_gray_value], dtype=np.uint8)
    damaged[crack_mask > 0] = crack_color

    return damaged


def generate_synthetic_dataset(clean_images_dir, output_dir, n_samples=20):
    """
    Generate a synthetic dataset of (clean, mask, damaged) triplets.
    As described in Section IV of arXiv 2602.12742.

    Args:
        clean_images_dir : folder with clean painting images
        output_dir       : folder to save generated data
        n_samples        : number of triplets to generate per image
    """
    os.makedirs(os.path.join(output_dir, "clean"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "damaged"), exist_ok=True)

    supported = (".jpg", ".jpeg", ".png")
    images = [f for f in os.listdir(clean_images_dir) if f.lower().endswith(supported)]

    if not images:
        print(f"No images found in {clean_images_dir}")
        return

    idx = 0
    for img_file in images:
        img_path = os.path.join(clean_images_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        # Resize to standard size (as in paper: ~598x375)
        image = cv2.resize(image, (598, 375))
        base = os.path.splitext(img_file)[0]

        for i in range(n_samples):
            crack_mask, _ = generate_crack_mask(image.shape)
            damaged = apply_crack_to_image(image, crack_mask)

            # Save triplet
            cv2.imwrite(os.path.join(output_dir, "clean",   f"{base}_{i:03d}.png"), image)
            cv2.imwrite(os.path.join(output_dir, "masks",   f"{base}_{i:03d}_mask.png"), crack_mask)
            cv2.imwrite(os.path.join(output_dir, "damaged", f"{base}_{i:03d}_damaged.png"), damaged)
            idx += 1

        print(f"  Generated {n_samples} pairs from: {img_file}")

    print(f"\nDataset generated: {idx} triplets in {output_dir}/")


def visualize_generation(image_path, output_dir="results"):
    """
    Visualize synthetic crack generation on a single image.
    Shows: original → crack mask → damaged image
    """
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    image = cv2.resize(image, (598, 375))
    crack_mask, raw_mask = generate_crack_mask(image.shape, n_cracks=100)
    damaged = apply_crack_to_image(image, crack_mask)

    base = os.path.splitext(os.path.basename(image_path))[0]

    # Save outputs
    cv2.imwrite(os.path.join(output_dir, f"{base}_synthetic_mask.png"), crack_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base}_synthetic_damaged.png"), damaged)

    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Clean Painting")
    axes[0].axis("off")

    axes[1].imshow(raw_mask, cmap="gray")
    axes[1].set_title("Raw Crack Mask\n(before refinement)")
    axes[1].axis("off")

    axes[2].imshow(crack_mask, cmap="gray")
    axes[2].set_title("Refined Crack Mask\n(after erosion + blur)")
    axes[2].axis("off")

    axes[3].imshow(cv2.cvtColor(damaged, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Synthetically Damaged\nPainting")
    axes[3].axis("off")

    plt.suptitle("Synthetic Craquelure Generation (arXiv 2602.12742)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base}_synthetic_generation.png"), dpi=150)
    plt.close()

    print(f"[SyntheticGen] Saved to {output_dir}/")
    return crack_mask, damaged


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_painting.jpg"
    visualize_generation(image_path)
    print("Synthetic crack generation complete.")
