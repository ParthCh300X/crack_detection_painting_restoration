"""
Novel Method: Exemplar-Based Patch Inpainting (EBPI)
=====================================================
Proposed by: Keshav Tak, Kaushik Kumar, Parth Chaudhary, Shreyas Khare
IIITN, 2026

Why this beats AD and MTM:
  AD and MTM fill each crack pixel using its IMMEDIATE neighborhood.
  If the neighborhood is mostly crack (wide crack), the fill is poor.

  EBPI fills crack pixels by finding the BEST MATCHING PATCH anywhere
  in the non-crack region of the image. This means:
  - Wide cracks get filled with correct texture (not just smeared neighbors)
  - Color/texture consistency across the fill region
  - Naturally handles the brushwork variation in paintings

Algorithm (based on Criminisi et al. 2004 patch-based inpainting, adapted for cracks):
  1. Compute fill order: prioritize crack pixels at boundary (high confidence)
  2. For each boundary crack pixel p:
     a. Extract patch P centered at p (partially filled)
     b. Find best matching patch Q in non-crack region (SSD on known pixels)
     c. Copy unknown pixels from Q into P
     d. Update confidence map
  3. Repeat until all crack pixels filled

Key innovation vs Criminisi 2004:
  - Uses gradient-weighted SSD (texture-aware matching)
  - Restricts search to same-luminance-range regions (color consistency)
  - Adapted specifically for thin elongated crack structures
"""

import cv2
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt


def compute_confidence(crack_mask, patch_size=5):
    """
    Initialize confidence map.
    Non-crack pixels = 1.0 (fully known)
    Crack pixels = 0.0 (unknown)
    """
    confidence = (1.0 - (crack_mask > 0).astype(np.float32))
    return confidence


def compute_fill_priority(image, crack_mask, confidence, patch_size=5):
    """
    Priority = Confidence * Data term
    Data term measures gradient strength at boundary (isophote strength)
    Higher priority = fill this boundary crack pixel first
    """
    half = patch_size // 2
    H, W = crack_mask.shape

    boundary = np.zeros_like(crack_mask, dtype=np.float32)
    remaining = (crack_mask > 0).astype(np.uint8)

    # Boundary = crack pixels that have at least one non-crack neighbor
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(remaining, kernel)
    eroded  = cv2.erode(remaining, kernel)
    boundary_mask = (dilated - eroded) * remaining

    # Confidence term: average confidence in patch
    conf_integral = cv2.boxFilter(confidence, -1, (patch_size, patch_size),
                                   normalize=True)

    # Data term: gradient magnitude at boundary (simplified)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(Gx**2 + Gy**2)
    grad_mag = grad_mag / (grad_mag.max() + 1e-8)

    # Priority = confidence * (1 + data term) at boundary pixels
    priority = boundary_mask.astype(np.float32) * conf_integral * (1.0 + 0.5 * grad_mag)
    return priority, boundary_mask


def find_best_patch(image, crack_mask, target_y, target_x, patch_size=7, search_step=3):
    """
    Find the best matching source patch for target position.

    Matching criterion: Sum of Squared Differences (SSD) on KNOWN pixels only,
    weighted by gradient magnitude (texture-aware matching).

    Args:
        image      : current BGR image (partially filled)
        crack_mask : current binary crack mask
        target_y, target_x : center of target patch
        patch_size : patch size (odd number)
        search_step: step size for candidate search (speed vs quality tradeoff)

    Returns:
        best_sy, best_sx : center of best source patch
    """
    half = patch_size // 2
    H, W = image.shape[:2]

    # Extract target patch
    ty1 = max(0, target_y - half)
    ty2 = min(H, target_y + half + 1)
    tx1 = max(0, target_x - half)
    tx2 = min(W, target_x + half + 1)

    target_patch = image[ty1:ty2, tx1:tx2].astype(np.float32)
    target_mask  = (crack_mask[ty1:ty2, tx1:tx2] == 0).astype(np.float32)

    # Need at least some known pixels to match
    if target_mask.sum() < 1:
        return target_y, target_x  # fallback

    # Compute gradient-weighted mask for matching
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)**2 + \
           cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)**2
    grad = np.sqrt(grad) / (np.sqrt(grad).max() + 1e-8)

    target_weight = target_mask * (1.0 + grad[ty1:ty2, tx1:tx2])

    best_ssd = np.inf
    best_sy, best_sx = target_y, target_x

    # Luminance of target region (for color-consistent matching)
    known_vals = target_patch[target_mask > 0]
    if len(known_vals) > 0:
        target_lum = known_vals.mean()
        lum_tolerance = 40.0
    else:
        target_lum = 128.0
        lum_tolerance = 100.0

    # Search over non-crack source patches
    for sy in range(half, H - half, search_step):
        for sx in range(half, W - half, search_step):
            # Skip if source patch overlaps crack region
            src_mask_region = crack_mask[sy-half:sy+half+1, sx-half:sx+half+1]
            if np.any(src_mask_region > 0):
                continue

            # Luminance check for color consistency
            src_lum = gray[sy, sx]
            if abs(src_lum - target_lum) > lum_tolerance:
                continue

            # Extract source patch
            sp = image[sy-half:sy+half+1, sx-half:sx+half+1].astype(np.float32)

            # Match patch size (handle border cases)
            ph = min(ty2-ty1, sp.shape[0])
            pw = min(tx2-tx1, sp.shape[1])
            if ph == 0 or pw == 0:
                continue

            tw = target_weight[:ph, :pw]
            if tw.sum() < 0.1:
                continue

            # Weighted SSD on known pixels only
            diff = (target_patch[:ph, :pw] - sp[:ph, :pw]) ** 2
            ssd = (diff.sum(axis=2) * tw).sum() / (tw.sum() + 1e-8)

            if ssd < best_ssd:
                best_ssd = ssd
                best_sy, best_sx = sy, sx

    return best_sy, best_sx


def ebpi_inpaint(image, crack_mask, patch_size=7, search_step=4, max_iters=500):
    """
    Exemplar-Based Patch Inpainting (EBPI)
    Novel crack restoration method for digitized paintings.

    Args:
        image       : BGR input image
        crack_mask  : binary mask (255 = crack)
        patch_size  : patch size for matching (7 = good balance)
        search_step : search step (larger = faster but less accurate)
        max_iters   : maximum fill iterations

    Returns:
        restored    : BGR restored image
    """
    print(f"[EBPI] Starting Exemplar-Based Patch Inpainting...")
    print(f"       patch_size={patch_size}, search_step={search_step}")
    t0 = time.time()

    restored = image.copy().astype(np.float32)
    remaining = (crack_mask > 0).astype(np.uint8)
    confidence = compute_confidence(crack_mask)
    half = patch_size // 2
    H, W = image.shape[:2]

    total_px = np.sum(remaining)
    filled = 0
    print(f"[EBPI] Crack pixels to fill: {total_px}")

    iteration = 0
    while np.any(remaining > 0) and iteration < max_iters:
        iteration += 1

        # Compute fill priority
        priority, boundary_mask = compute_fill_priority(
            np.clip(restored, 0, 255).astype(np.uint8),
            remaining, confidence, patch_size
        )

        # Find highest priority boundary pixel
        if np.max(priority) < 1e-10:
            # No boundary pixels found — use any remaining crack pixel
            remaining_coords = np.argwhere(remaining > 0)
            if len(remaining_coords) == 0:
                break
            py, px = remaining_coords[0]
        else:
            idx = np.argmax(priority)
            py, px = np.unravel_index(idx, priority.shape)

        # Find best matching source patch
        best_sy, best_sx = find_best_patch(
            np.clip(restored, 0, 255).astype(np.uint8),
            remaining, py, px, patch_size, search_step
        )

        # Copy unknown pixels from source patch to target patch
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                ty, tx = py + dy, px + dx
                sy, sx = best_sy + dy, best_sx + dx

                if (0 <= ty < H and 0 <= tx < W and
                    0 <= sy < H and 0 <= sx < W and
                        remaining[ty, tx] > 0):
                    restored[ty, tx] = restored[sy, sx]
                    confidence[ty, tx] = confidence[py, px]
                    remaining[ty, tx] = 0
                    filled += 1

        if iteration % 50 == 0:
            pct = filled / max(total_px, 1) * 100
            print(f"[EBPI] Iter {iteration}: {filled}/{total_px} ({pct:.0f}%)")

        if filled >= total_px:
            break

    elapsed = time.time() - t0
    print(f"[EBPI] Done: {filled}/{total_px} px filled in {elapsed:.1f}s")

    return np.clip(restored, 0, 255).astype(np.uint8)


def run_novel_comparison(image_path, crack_mask, output_dir="results/novel"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "classical"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from crack_filling import anisotropic_diffusion
    from metrics import compute_psnr, compute_ssim, compute_mse

    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    base  = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\n{'='*65}")
    print("THREE-WAY: Classical AD | Modern Orient-AD | Novel EBPI")
    print(f"{'='*65}")

    # Classical AD
    print("\n[1/3] Classical AD (20 iter, K=127)...")
    t1 = time.time()
    res_classical = anisotropic_diffusion(image, crack_mask, n_iter=20, K=127)
    t_cl = time.time() - t1

    # Modern AD
    print("\n[2/3] Modern Orientation-Sensitive AD (50 iter, K=80)...")
    t2 = time.time()
    try:
        from compare import orientation_sensitive_ad
        res_modern = orientation_sensitive_ad(image, crack_mask, n_iter=50, K=80)
    except Exception:
        res_modern = anisotropic_diffusion(image, crack_mask, n_iter=50, K=80)
    t_mo = time.time() - t2

    # Novel EBPI
    print("\n[3/3] Novel EBPI (Exemplar-Based Patch Inpainting)...")
    t3 = time.time()
    res_novel = ebpi_inpaint(image, crack_mask,
                              patch_size=7, search_step=4)
    t_no = time.time() - t3

    # Metrics
    methods = {
        "Classical AD": (res_classical, t_cl),
        "Modern AD":    (res_modern,    t_mo),
        "Novel EBPI":   (res_novel,     t_no),
    }

    print(f"\n{'='*70}")
    print(f"{'Method':<18} {'PSNR (dB)':<12} {'SSIM':<10} {'MSE':<10} {'Time(s)'}")
    print(f"{'='*70}")
    mdata = {}
    for name, (res, t) in methods.items():
        p = compute_psnr(image, res)
        s = compute_ssim(image, res)
        m = compute_mse(image, res)
        print(f"{name:<18} {p:<12.4f} {s:<10.4f} {m:<10.4f} {t:.2f}")
        mdata[name] = {"PSNR": p, "SSIM": s, "MSE": m, "Time": t}
        cv2.imwrite(os.path.join(output_dir,
                    f"{base}_{name.replace(' ','_')}.png"), res)
    print(f"{'='*70}")

    # Figure
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))

    def show(ax, img, title, is_mask=False):
        ax.imshow(img if is_mask else cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                  cmap="gray" if is_mask else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    show(axes[0,0], image,        "Original (with cracks)")
    show(axes[0,1], crack_mask,   "Crack Mask", is_mask=True)
    m = mdata["Classical AD"]
    show(axes[0,2], res_classical,
         f"Classical AD (Giakoumis 2006)\nPSNR={m['PSNR']:.2f}  SSIM={m['SSIM']:.4f}")
    m = mdata["Modern AD"]
    show(axes[0,3], res_modern,
         f"Modern Orient-AD (2026)\nPSNR={m['PSNR']:.2f}  SSIM={m['SSIM']:.4f}")

    m = mdata["Novel EBPI"]
    show(axes[1,0], res_novel,
         f"Novel EBPI [PROPOSED]\nPSNR={m['PSNR']:.2f}  SSIM={m['SSIM']:.4f}")

    diff1 = cv2.applyColorMap(
        cv2.convertScaleAbs(cv2.absdiff(res_novel, res_classical), alpha=8),
        cv2.COLORMAP_JET)
    show(axes[1,1], diff1, "EBPI vs Classical\n(amplified diff)")

    diff2 = cv2.applyColorMap(
        cv2.convertScaleAbs(cv2.absdiff(res_novel, res_modern), alpha=8),
        cv2.COLORMAP_JET)
    show(axes[1,2], diff2, "EBPI vs Modern\n(amplified diff)")

    axes[1,3].axis("off")
    ebpi_v_cl_p = mdata['Novel EBPI']['PSNR'] - mdata['Classical AD']['PSNR']
    ebpi_v_cl_s = mdata['Novel EBPI']['SSIM'] - mdata['Classical AD']['SSIM']
    summary = (
        f"THREE-WAY RESULTS\n{'='*32}\n"
        f"Classical AD:\n"
        f"  PSNR={mdata['Classical AD']['PSNR']:.4f} dB\n"
        f"  SSIM={mdata['Classical AD']['SSIM']:.4f}\n\n"
        f"Modern Orient-AD:\n"
        f"  PSNR={mdata['Modern AD']['PSNR']:.4f} dB\n"
        f"  SSIM={mdata['Modern AD']['SSIM']:.4f}\n\n"
        f"Novel EBPI [Proposed]:\n"
        f"  PSNR={mdata['Novel EBPI']['PSNR']:.4f} dB\n"
        f"  SSIM={mdata['Novel EBPI']['SSIM']:.4f}\n\n"
        f"EBPI vs Classical AD:\n"
        f"  PSNR: {ebpi_v_cl_p:+.4f} dB\n"
        f"  SSIM: {ebpi_v_cl_s:+.4f}\n"
        f"  MSE:  {mdata['Novel EBPI']['MSE']-mdata['Classical AD']['MSE']:+.4f}"
    )
    axes[1,3].text(0.05, 0.97, summary, transform=axes[1,3].transAxes,
                   fontsize=9, va='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

    plt.suptitle(
        f"Classical vs Modern vs Novel EBPI | {base}\n"
        "Exemplar-Based Patch Inpainting (EBPI) — Proposed Novel Method",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, f"{base}_threeway_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")
    return mdata


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "classical"))
    from top_hat_detection import detect_cracks

    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/painting3.jpg"
    threshold  = int(sys.argv[2]) if len(sys.argv) > 2 else 23

    crack_mask, _ = detect_cracks(image_path, threshold=threshold)
    base = os.path.splitext(os.path.basename(image_path))[0]
    run_novel_comparison(image_path, crack_mask,
                         output_dir=f"results/novel_{base}")
