"""
Full Modern Pipeline (arXiv 2602.12742 inspired)
Combines:
  1. Top-hat detection
  2. Synthetic crack generation (for demo/training data)
  3. SegFormer zero-shot refinement (or classical fallback)
  4. Anisotropic Diffusion inpainting

Usage:
    python modern_pipeline.py <image_path> [--mode auto|segformer|classical]
"""

import cv2
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "classical"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.insert(0, os.path.dirname(__file__))

from top_hat_detection import detect_cracks
from crack_filling import anisotropic_diffusion, mtm_filter
from metrics import compute_psnr, compute_ssim, compute_mse
from synthetic_crack_generator import visualize_generation, generate_crack_mask, apply_crack_to_image
from segformer_refinement import refine_crack_mask


def run_modern_pipeline(image_path, refinement_mode="auto", threshold=23, output_dir=None):
    """
    Full modern pipeline as described in arXiv 2602.12742.

    Pipeline:
    Input → Top-Hat Detection → Size Filtering → SegFormer Refinement
                                                          ↓
                                              Anisotropic Diffusion
                                                          ↓
                                              Restored Painting

    Args:
        image_path      : path to painting with real cracks
        refinement_mode : "auto" | "segformer" | "classical"
        threshold       : top-hat threshold
        output_dir      : save directory

    Returns:
        results dict with all intermediate outputs and metrics
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    if output_dir is None:
        output_dir = f"results/{base}_modern"
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    print(f"\n{'='*65}")
    print(f"MODERN PIPELINE (arXiv 2602.12742) | Image: {base}")
    print(f"{'='*65}")

    # ── Step 1: Initial crack detection via top-hat ──────────────────
    print("\n[1/4] Top-hat crack detection + size-based noise filtering...")
    crack_mask_raw, luminance = detect_cracks(
        image_path,
        threshold=threshold,
        output_dir=output_dir
    )
    raw_count = np.sum(crack_mask_raw > 0)
    print(f"      Initial crack candidates: {raw_count} pixels")

    # ── Step 2: Refinement (SegFormer zero-shot or classical HSV) ────
    print(f"\n[2/4] Mask refinement ({refinement_mode} mode)...")
    crack_mask_refined = refine_crack_mask(
        crack_mask_raw,
        image,
        mode=refinement_mode,
        output_dir=output_dir
    )
    refined_count = np.sum(crack_mask_refined > 0)
    cv2.imwrite(os.path.join(output_dir, f"{base}_refined_mask.png"), crack_mask_refined)

    # ── Step 3: Crack filling via Anisotropic Diffusion ──────────────
    print("\n[3/4] Crack filling via Anisotropic Diffusion (20 iterations)...")
    restored_ad = anisotropic_diffusion(image, crack_mask_refined, n_iter=20, lambda_=0.25, K=127)
    cv2.imwrite(os.path.join(output_dir, f"{base}_modern_restored_AD.png"), restored_ad)

    # Also run MTM for comparison
    print("      Also running MTM filter for comparison...")
    restored_mtm = mtm_filter(image, crack_mask_refined)
    cv2.imwrite(os.path.join(output_dir, f"{base}_modern_restored_MTM.png"), restored_mtm)

    # ── Step 4: Metrics ──────────────────────────────────────────────
    print("\n[4/4] Computing metrics...")
    metrics = {
        "AD":  {"PSNR": compute_psnr(image, restored_ad),
                "SSIM": compute_ssim(image, restored_ad),
                "MSE":  compute_mse(image, restored_ad)},
        "MTM": {"PSNR": compute_psnr(image, restored_mtm),
                "SSIM": compute_ssim(image, restored_mtm),
                "MSE":  compute_mse(image, restored_mtm)},
    }

    print(f"\n{'─'*50}")
    print(f"{'Method':<8} {'PSNR (dB)':<12} {'SSIM':<10} {'MSE':<10}")
    print(f"{'─'*50}")
    for method, m in metrics.items():
        print(f"{method:<8} {m['PSNR']:<12.4f} {m['SSIM']:<10.4f} {m['MSE']:<10.4f}")
    print(f"{'─'*50}")
    print(f"Raw crack pixels   : {raw_count}")
    print(f"Refined crack pixels: {refined_count}")
    print(f"False positives removed: {raw_count - refined_count} "
          f"({(raw_count - refined_count)/max(raw_count,1)*100:.1f}%)")

    # ── Final comparison visualization ──────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("(1) Original (with cracks)", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(crack_mask_raw, cmap="gray")
    axes[0, 1].set_title(f"(2) Initial Top-Hat Mask\n({raw_count} px)", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(crack_mask_refined, cmap="gray")
    axes[0, 2].set_title(f"(3) Refined Mask\n({refined_count} px, {(raw_count-refined_count)/max(raw_count,1)*100:.0f}% FP removed)", fontsize=11)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Original (reference)", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(restored_mtm, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"(4a) MTM Restored\nPSNR={metrics['MTM']['PSNR']:.2f} SSIM={metrics['MTM']['SSIM']:.3f}", fontsize=11)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(cv2.cvtColor(restored_ad, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f"(4b) AD Restored (Best)\nPSNR={metrics['AD']['PSNR']:.2f} SSIM={metrics['AD']['SSIM']:.3f}", fontsize=11)
    axes[1, 2].axis("off")

    plt.suptitle(f"Modern Pipeline (arXiv 2602.12742) | {base}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base}_modern_full_pipeline.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Modern pipeline complete. Results in: {output_dir}/")
    return metrics, crack_mask_refined, restored_ad


def demo_synthetic_generation(image_path, output_dir="results/synthetic_demo"):
    """
    Demonstrate synthetic crack generation on a clean painting.
    Shows the key contribution of the 2026 paper.
    """
    print(f"\n{'='*65}")
    print("SYNTHETIC CRACK GENERATION DEMO (Section IV of arXiv 2602.12742)")
    print(f"{'='*65}")

    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    image_resized = cv2.resize(image, (598, 375))

    # Generate 3 different crack patterns on same image
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Clean Painting\n(input)", fontsize=10)
    axes[0, 0].axis("off")

    for i in range(3):
        n_cracks = np.random.randint(60, 130)
        mask, _ = generate_crack_mask(image_resized.shape, n_cracks=n_cracks)
        damaged = apply_crack_to_image(image_resized, mask)

        axes[0, i+1].imshow(mask, cmap="gray")
        axes[0, i+1].set_title(f"Synthetic Mask #{i+1}\n({n_cracks} crack trajectories)", fontsize=10)
        axes[0, i+1].axis("off")

        axes[1, i+1].imshow(cv2.cvtColor(damaged, cv2.COLOR_BGR2RGB))
        axes[1, i+1].set_title(f"Damaged Image #{i+1}\n(training sample)", fontsize=10)
        axes[1, i+1].axis("off")

        cv2.imwrite(os.path.join(output_dir, f"synthetic_mask_{i+1}.png"), mask)
        cv2.imwrite(os.path.join(output_dir, f"synthetic_damaged_{i+1}.png"), damaged)

    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5,
        "Bezier Trajectories\n+ Tapered Geometry\n+ Branching\n→ Realistic Cracks",
        ha="center", va="center", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.suptitle("Synthetic Craquelure Generation — Bezier Curve Approach\n(arXiv:2602.12742, Feb 2026)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "synthetic_generation_overview.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Synthetic generation demo saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modern crack detection & restoration pipeline")
    parser.add_argument("image", nargs="?", default="data/sample_painting.jpg")
    parser.add_argument("--mode", choices=["auto", "segformer", "classical"], default="auto")
    parser.add_argument("--threshold", type=int, default=23)
    parser.add_argument("--demo-synth", action="store_true",
                        help="Also run synthetic generation demo")
    args = parser.parse_args()

    if args.demo_synth:
        demo_synthetic_generation(args.image)

    run_modern_pipeline(args.image, refinement_mode=args.mode, threshold=args.threshold)
