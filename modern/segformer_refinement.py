"""
Crack Mask Refinement Module
Based on: Cuch-Guillen et al., arXiv:2602.12742, Feb 2026

Uses shape-geometry (elongation) + color (HSV) filtering
to remove false positives while retaining genuine crack pixels.
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def compute_elongation(contour):
    """Elongation = major_axis / minor_axis via fitted ellipse."""
    if len(contour) < 5:
        x, y, w, h = cv2.boundingRect(contour)
        return max(w, h) / max(min(w, h), 1)
    try:
        ellipse = cv2.fitEllipse(contour)
        major = max(ellipse[1])
        minor = max(min(ellipse[1]), 0.1)
        return major / minor
    except:
        x, y, w, h = cv2.boundingRect(contour)
        return max(w, h) / max(min(w, h), 1)


def shape_based_refinement(crack_mask, image,
                            min_elongation=1.5,
                            min_area=5,
                            max_area_ratio=0.03):
    """
    Remove false positives using shape geometry.
    Genuine cracks: elongated (high aspect ratio)
    False positives (brush strokes, noise): round/blobby

    Tuned params:
    - min_elongation=1.5  (was 2.0 — too aggressive, removed real cracks)
    - max_area_ratio=0.03 (was 0.015 — allows slightly larger crack regions)
    """
    H, W = crack_mask.shape[:2]
    max_area = int(H * W * max_area_ratio)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        crack_mask, connectivity=8
    )

    refined_mask = np.zeros_like(crack_mask)
    kept = 0
    removed = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        # Remove noise
        if area < min_area:
            removed += 1
            continue

        # Remove huge blobs (non-crack dark regions)
        if area > max_area:
            removed += 1
            continue

        component = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            removed += 1
            continue

        contour = max(contours, key=cv2.contourArea)
        elongation = compute_elongation(contour)

        if elongation >= min_elongation:
            refined_mask[labels == label] = 255
            kept += 1
        else:
            removed += 1

    return refined_mask, kept, removed


def color_based_refinement(crack_mask, image):
    """
    HSV color filtering.
    Cracks: warm hue (0-60 deg), moderate saturation.
    Very dark pixels always pass regardless of hue.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.float32)        # [0, 179] in OpenCV
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    V = hsv[:, :, 2].astype(np.float32) / 255.0

    # Very dark pixels = definite cracks
    very_dark = (V < 0.20)

    # Warm-toned cracks (hue 0-40 in OpenCV = 0-80 degrees)
    crack_hue = (H <= 45) | (H >= 155)
    crack_sat = (S >= 0.05) & (S <= 0.95)

    color_valid = (crack_hue & crack_sat) | very_dark
    color_valid = color_valid.astype(np.uint8) * 255

    refined = cv2.bitwise_and(crack_mask, color_valid)
    return refined


def advanced_refinement(crack_mask, image, output_dir="results"):
    """
    Full refinement pipeline:
    1. Shape-based (elongation) filtering
    2. Color-based (HSV) filtering
    3. Morphological cleanup

    Tuning target: remove 30-60% of pixels as false positives.
    Removing 90%+ means we're being too aggressive and losing real cracks.
    """
    os.makedirs(output_dir, exist_ok=True)

    before_count = np.sum(crack_mask > 0)
    print(f"[Refinement] Input crack pixels: {before_count}")

    # ── Stage 1: Shape filtering ─────────────────────────────────────
    print("[Refinement] Stage 1: Shape geometry analysis (elongation filtering)...")
    shape_refined, kept, removed_shape = shape_based_refinement(
        crack_mask, image,
        min_elongation=1.5,    # tuned down from 2.0 — less aggressive
        min_area=8,
        max_area_ratio=0.03    # tuned up from 0.015 — allow larger crack regions
    )
    shape_count = np.sum(shape_refined > 0)
    print(f"             Kept: {kept} components | Removed: {removed_shape} blobs")
    print(f"             Pixels: {before_count} -> {shape_count} "
          f"({(before_count-shape_count)/max(before_count,1)*100:.1f}% removed)")

    # Safety check: if Stage 1 removed more than 70%, it's too aggressive
    # Fall back to just size-based noise removal
    if shape_count < 0.30 * before_count:
        print("[Refinement] Stage 1 too aggressive — using size-only fallback")
        # Only remove tiny noise components (< 5 px) and huge blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crack_mask, 8)
        H, W = crack_mask.shape[:2]
        max_area = int(H * W * 0.05)
        shape_refined = np.zeros_like(crack_mask)
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if 5 <= area <= max_area:
                shape_refined[labels == lbl] = 255
        shape_count = np.sum(shape_refined > 0)
        print(f"             Fallback result: {shape_count} pixels")

    # ── Stage 2: Color filtering ──────────────────────────────────────
    print("[Refinement] Stage 2: HSV color filtering...")
    color_refined = color_based_refinement(shape_refined, image)
    color_count = np.sum(color_refined > 0)
    print(f"             Pixels: {shape_count} -> {color_count} "
          f"({(shape_count-color_count)/max(shape_count,1)*100:.1f}% removed)")

    # ── Stage 3: Morphological cleanup ───────────────────────────────
    print("[Refinement] Stage 3: Morphological cleanup...")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    final_mask = cv2.morphologyEx(color_refined, cv2.MORPH_OPEN, kernel)

    # Remove remaining tiny noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, 8)
    clean_mask = np.zeros_like(final_mask)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 5:
            clean_mask[labels == lbl] = 255
    final_mask = clean_mask

    final_count = np.sum(final_mask > 0)
    total_removed = before_count - final_count
    total_pct = total_removed / max(before_count, 1) * 100

    # ── Final safety: if we removed too much, revert to shape-only ───
    if final_count < 0.25 * before_count:
        print("[Refinement] WARNING: Final mask too sparse, reverting to shape-refined mask")
        final_mask = shape_refined.copy()
        final_count = np.sum(final_mask > 0)
        total_removed = before_count - final_count
        total_pct = total_removed / max(before_count, 1) * 100

    print(f"\n[Refinement] SUMMARY:")
    print(f"             Before : {before_count} pixels")
    print(f"             After  : {final_count} pixels")
    print(f"             Removed: {total_removed} pixels ({total_pct:.1f}% false positives)")

    # ── Save intermediate outputs ─────────────────────────────────────
    cv2.imwrite(os.path.join(output_dir, "mask_raw.png"), crack_mask)
    cv2.imwrite(os.path.join(output_dir, "mask_shape_refined.png"), shape_refined)
    cv2.imwrite(os.path.join(output_dir, "mask_final_refined.png"), final_mask)

    # ── Visualization ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(crack_mask, cmap="gray")
    axes[1].set_title(f"Raw Top-Hat Mask\n({before_count} px)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(shape_refined, cmap="gray")
    axes[2].set_title(f"After Shape Filtering\n({shape_count} px, "
                      f"{(before_count-shape_count)/max(before_count,1)*100:.0f}% removed)", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(final_mask, cmap="gray")
    axes[3].set_title(f"Final Refined Mask\n({final_count} px, {total_pct:.0f}% FP removed)", fontsize=10)
    axes[3].axis("off")

    plt.suptitle("Mask Refinement: Shape + Color Based False Positive Removal", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "refinement_stages.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return final_mask


# ── Public API ────────────────────────────────────────────────────────────

def refine_crack_mask(crack_mask, image, mode="auto", output_dir="results"):
    """Main entry point. mode param kept for API compatibility."""
    return advanced_refinement(crack_mask, image, output_dir=output_dir)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "classical"))
    from top_hat_detection import detect_cracks

    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/painting1.jpg"
    crack_mask, _ = detect_cracks(image_path, threshold=23)
    image = cv2.imread(image_path)
    refined = refine_crack_mask(crack_mask, image, output_dir="results/refinement_test")
    print(f"\nFinal crack pixels: {np.sum(refined > 0)}")
