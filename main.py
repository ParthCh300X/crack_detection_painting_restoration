"""
Main Pipeline: Crack Detection and Removal in Digitized Paintings
Runs full classical pipeline on all images in data/ folder.

Usage:
    python main.py                          # runs on all images in data/
    python main.py data/my_painting.jpg     # runs on single image
"""

import os
import sys
import cv2
import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "classical"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

from top_hat_detection import detect_cracks
from crack_filling import fill_cracks
from metrics import compute_psnr, compute_ssim, compute_mse


def run_pipeline(image_path, threshold=23):
    """Run full classical pipeline on a single image."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = f"results/{base}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")

    # Step 1: Detect cracks
    print("[1/3] Detecting cracks via top-hat transform...")
    crack_mask, luminance = detect_cracks(image_path, threshold=threshold, output_dir=out_dir)
    print(f"      Crack pixels: {np.sum(crack_mask > 0)}")

    # Step 2: Fill cracks - MTM
    print("[2/3] Filling cracks with MTM filter...")
    restored_mtm = fill_cracks(image_path, crack_mask, method="MTM", output_dir=out_dir)

    # Step 3: Fill cracks - Anisotropic Diffusion
    print("[3/3] Filling cracks with Anisotropic Diffusion...")
    restored_ad = fill_cracks(image_path, crack_mask, method="AD", output_dir=out_dir)

    # Metrics
    original = cv2.imread(image_path)
    print(f"\n--- Metrics vs Original ---")
    print(f"MTM  | PSNR: {compute_psnr(original, restored_mtm):.2f} dB | "
          f"SSIM: {compute_ssim(original, restored_mtm):.4f} | "
          f"MSE: {compute_mse(original, restored_mtm):.2f}")
    print(f"AD   | PSNR: {compute_psnr(original, restored_ad):.2f} dB  | "
          f"SSIM: {compute_ssim(original, restored_ad):.4f} | "
          f"MSE: {compute_mse(original, restored_ad):.2f}")

    return crack_mask, restored_mtm, restored_ad


def main():
    if len(sys.argv) > 1:
        # Single image mode
        image_path = sys.argv[1]
        threshold  = int(sys.argv[2]) if len(sys.argv) > 2 else 23
        run_pipeline(image_path, threshold=threshold)
    else:
        # Batch mode — run on all images in data/
        data_dir = "data"
        supported = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        images = [f for f in os.listdir(data_dir) if f.lower().endswith(supported)]

        if not images:
            print("No images found in data/ folder.")
            print("Add .jpg or .png painting images to the data/ directory.")
            return

        print(f"Found {len(images)} image(s) in data/")
        for img_file in images:
            run_pipeline(os.path.join(data_dir, img_file))

    print(f"\n✅ All done! Results saved in results/ folder.")


if __name__ == "__main__":
    main()
