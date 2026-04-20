"""
Evaluation Metrics for Crack Restoration
- PSNR  : Peak Signal-to-Noise Ratio
- SSIM  : Structural Similarity Index
- F1    : For crack detection evaluation (if ground truth mask available)
- MSE   : Mean Squared Error
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_psnr(original, restored):
    """Higher is better. >30 dB is generally good."""
    orig = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(np.float64)
    rest = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB).astype(np.float64)
    return psnr(orig, rest, data_range=255)


def compute_ssim(original, restored):
    """Range [0,1]. Higher is better. >0.8 is generally good."""
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    rest_gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(orig_gray, rest_gray, full=True)
    return score


def compute_mse(original, restored):
    """Lower is better."""
    orig = original.astype(np.float64)
    rest = restored.astype(np.float64)
    return np.mean((orig - rest) ** 2)


def compute_f1(pred_mask, gt_mask):
    """
    F1 score for crack detection.
    pred_mask, gt_mask: binary numpy arrays (0 or 255)
    """
    pred = (pred_mask > 0).astype(np.uint8).flatten()
    gt   = (gt_mask > 0).astype(np.uint8).flatten()

    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_restoration(original_path, restored_mtm_path, restored_ad_path):
    """
    Compare MTM vs AD restoration results.
    Note: Since we have no ground truth crack-free image,
    we use the original (cracked) image as reference — 
    metrics show how much the image changed (lower change = more conservative).
    For proper eval, a synthetic cracked image + clean original is needed.
    """
    orig = cv2.imread(original_path)
    mtm  = cv2.imread(restored_mtm_path)
    ad   = cv2.imread(restored_ad_path)

    if orig is None or mtm is None or ad is None:
        print("Could not load one or more images for evaluation.")
        return

    print("\n" + "="*50)
    print("RESTORATION EVALUATION METRICS")
    print("="*50)

    print("\n[MTM Filter Results]")
    print(f"  PSNR : {compute_psnr(orig, mtm):.4f} dB")
    print(f"  SSIM : {compute_ssim(orig, mtm):.4f}")
    print(f"  MSE  : {compute_mse(orig, mtm):.4f}")

    print("\n[Anisotropic Diffusion Results]")
    print(f"  PSNR : {compute_psnr(orig, ad):.4f} dB")
    print(f"  SSIM : {compute_ssim(orig, ad):.4f}")
    print(f"  MSE  : {compute_mse(orig, ad):.4f}")
    print("="*50)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        evaluate_restoration(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python metrics.py <original> <restored_MTM> <restored_AD>")
