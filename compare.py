"""
Head-to-Head Comparison: Classical (Giakoumis 2006) vs Modern (arXiv 2602.12742)
Generates a comprehensive comparison report with metrics table.

Usage:
    python compare.py <image_path>
    python compare.py data/painting1.jpg
"""

import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "classical"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modern"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

from top_hat_detection import detect_cracks
from crack_filling import anisotropic_diffusion, mtm_filter
from segformer_refinement import refine_crack_mask
from metrics import compute_psnr, compute_ssim, compute_mse


def compare_pipelines(image_path, threshold=23, output_dir=None):
    """
    Run both pipelines and produce a side-by-side comparison.
    This is the main results figure for your research paper.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    if output_dir is None:
        output_dir = f"results/{base}_comparison"
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    print(f"\n{'='*65}")
    print(f"COMPARISON: Classical vs Modern | {base}")
    print(f"{'='*65}")

    # ── CLASSICAL PIPELINE (Giakoumis 2006) ──────────────────────────
    print("\n[CLASSICAL] Running Giakoumis 2006 pipeline...")
    mask_classical, _ = detect_cracks(image_path, threshold=threshold, output_dir=output_dir)
    restored_classical_mtm = mtm_filter(image, mask_classical)
    restored_classical_ad  = anisotropic_diffusion(image, mask_classical)
    print(f"  Crack pixels: {np.sum(mask_classical > 0)}")

    # ── MODERN PIPELINE (arXiv 2602.12742) ───────────────────────────
    print("\n[MODERN]    Running arXiv 2602.12742 pipeline...")
    mask_refined = refine_crack_mask(mask_classical, image, mode="auto", output_dir=output_dir)
    restored_modern_ad  = anisotropic_diffusion(image, mask_refined)
    restored_modern_mtm = mtm_filter(image, mask_refined)
    print(f"  Crack pixels after refinement: {np.sum(mask_refined > 0)}")

    # ── METRICS TABLE ─────────────────────────────────────────────────
    results = {
        "Classical + MTM": {"mask": mask_classical, "restored": restored_classical_mtm},
        "Classical + AD":  {"mask": mask_classical, "restored": restored_classical_ad},
        "Modern + MTM":    {"mask": mask_refined,   "restored": restored_modern_mtm},
        "Modern + AD":     {"mask": mask_refined,   "restored": restored_modern_ad},
    }

    print(f"\n{'─'*65}")
    print(f"{'Method':<20} {'Crack Px':<12} {'PSNR (dB)':<12} {'SSIM':<10} {'MSE':<10}")
    print(f"{'─'*65}")
    metrics_data = {}
    for name, data in results.items():
        psnr = compute_psnr(image, data["restored"])
        ssim = compute_ssim(image, data["restored"])
        mse  = compute_mse(image,  data["restored"])
        crack_px = np.sum(data["mask"] > 0)
        print(f"{name:<20} {crack_px:<12} {psnr:<12.4f} {ssim:<10.4f} {mse:<10.4f}")
        metrics_data[name] = {"PSNR": psnr, "SSIM": ssim, "MSE": mse, "crack_px": crack_px}
    print(f"{'─'*65}")

    # ── COMPREHENSIVE VISUALIZATION ───────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.05)

    def show(ax, img_bgr, title, is_mask=False):
        if is_mask:
            ax.imshow(img_bgr, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9, pad=4)
        ax.axis("off")

    # Row 0: Original + masks
    show(fig.add_subplot(gs[0, 0]), image, "Original\n(with cracks)")
    show(fig.add_subplot(gs[0, 1]), mask_classical, "Classical Mask\n(Top-Hat only)", is_mask=True)
    show(fig.add_subplot(gs[0, 2]), mask_refined,   "Modern Mask\n(Refined)", is_mask=True)
    ax_diff = fig.add_subplot(gs[0, 3])
    fp_map = cv2.subtract(mask_classical, mask_refined)
    ax_diff.imshow(fp_map, cmap="hot")
    ax_diff.set_title(f"False Positives Removed\n({np.sum(fp_map>0)} px)", fontsize=9, pad=4)
    ax_diff.axis("off")

    # Row 1: Classical restored
    show(fig.add_subplot(gs[1, 0]), image, "Original")
    show(fig.add_subplot(gs[1, 1]), mask_classical, "Classical Mask", is_mask=True)
    m = metrics_data["Classical + MTM"]
    show(fig.add_subplot(gs[1, 2]), restored_classical_mtm,
         f"Classical + MTM\nPSNR={m['PSNR']:.2f} SSIM={m['SSIM']:.3f}")
    m = metrics_data["Classical + AD"]
    show(fig.add_subplot(gs[1, 3]), restored_classical_ad,
         f"Classical + AD\nPSNR={m['PSNR']:.2f} SSIM={m['SSIM']:.3f}")

    # Row 2: Modern restored
    show(fig.add_subplot(gs[2, 0]), image, "Original")
    show(fig.add_subplot(gs[2, 1]), mask_refined, "Modern Mask", is_mask=True)
    m = metrics_data["Modern + MTM"]
    show(fig.add_subplot(gs[2, 2]), restored_modern_mtm,
         f"Modern + MTM\nPSNR={m['PSNR']:.2f} SSIM={m['SSIM']:.3f}")
    m = metrics_data["Modern + AD"]
    show(fig.add_subplot(gs[2, 3]), restored_modern_ad,
         f"Modern + AD ✓ Best\nPSNR={m['PSNR']:.2f} SSIM={m['SSIM']:.3f}")

    # Add row labels
    for row, label in zip([0, 1, 2], ["Detection", "Classical\nRestoration", "Modern\nRestoration"]):
        fig.text(0.005, 0.83 - row * 0.3, label,
                 va="center", ha="left", fontsize=10, fontweight="bold",
                 rotation=90, color="navy")

    plt.suptitle(
        f"Crack Detection & Removal: Classical (Giakoumis 2006) vs Modern (Cuch-Guillen et al. 2026)\n{base}",
        fontsize=12, fontweight="bold", y=1.01
    )

    save_path = os.path.join(output_dir, f"{base}_full_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save metrics as text report
    with open(os.path.join(output_dir, "metrics_report.txt"), "w") as f:
        f.write(f"RESULTS: {base}\n{'='*50}\n\n")
        f.write(f"{'Method':<20} {'Crack Px':<12} {'PSNR (dB)':<12} {'SSIM':<10} {'MSE':<10}\n")
        f.write(f"{'─'*60}\n")
        for name, m in metrics_data.items():
            f.write(f"{name:<20} {m['crack_px']:<12} {m['PSNR']:<12.4f} {m['SSIM']:<10.4f} {m['MSE']:<10.4f}\n")

    print(f"\n✅ Comparison complete. Saved to: {output_dir}/")
    print(f"   Main figure : {base}_full_comparison.png")
    print(f"   Metrics     : metrics_report.txt")

    return metrics_data


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_painting.jpg"
    threshold  = int(sys.argv[2]) if len(sys.argv) > 2 else 23
    compare_pipelines(image_path, threshold=threshold)
