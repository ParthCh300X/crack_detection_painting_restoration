"""
Crack Filling Methods
1. Modified Trimmed Mean (MTM) Filter
2. Anisotropic Diffusion (AD)

Based on: Giakoumis et al., IEEE TIP 2006
          Cuch-Guillen et al., arXiv 2602.12742, Feb 2026
"""

import cv2
import numpy as np
import os


# ─────────────────────────────────────────────
# METHOD 1: Modified Trimmed Mean (MTM) Filter
# ─────────────────────────────────────────────

def mtm_filter(image, crack_mask, window_size=5):
    """
    Fill crack pixels using the Modified Trimmed Mean filter.
    Averages only NON-crack pixels in the local window.
    Processes outer-to-inner (boundary pixels first) to avoid error propagation.

    Args:
        image       : input BGR image
        crack_mask  : binary mask (255 = crack pixel)
        window_size : filter window size (should be >= crack width + 1)

    Returns:
        restored    : BGR image with cracks filled
    """
    restored = image.copy().astype(np.float32)
    mask = (crack_mask > 0).astype(np.uint8)
    half = window_size // 2

    # Work per channel independently
    for c in range(3):
        channel = restored[:, :, c].copy()
        remaining = mask.copy()

        max_iters = 500
        iteration = 0

        while np.any(remaining > 0) and iteration < max_iters:
            iteration += 1
            filled_this_round = False

            crack_pixels = np.argwhere(remaining > 0)
            updates = {}

            for (y, x) in crack_pixels:
                # Extract window
                y1, y2 = max(0, y - half), min(channel.shape[0], y + half + 1)
                x1, x2 = max(0, x - half), min(channel.shape[1], x + half + 1)

                win_img  = channel[y1:y2, x1:x2]
                win_mask = remaining[y1:y2, x1:x2]

                # Only use non-crack pixels
                non_crack_vals = win_img[win_mask == 0]

                if len(non_crack_vals) > 0:
                    updates[(y, x)] = np.mean(non_crack_vals)
                    filled_this_round = True

            # Apply updates
            for (y, x), val in updates.items():
                channel[y, x] = val
                remaining[y, x] = 0

            restored[:, :, c] = channel

            if not filled_this_round:
                break

    return np.clip(restored, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# METHOD 2: Anisotropic Diffusion (AD)
# ─────────────────────────────────────────────

def anisotropic_diffusion(image, crack_mask, n_iter=20, lambda_=0.25, K=127):
    """
    Fill cracks using Perona-Malik anisotropic diffusion.
    Diffusion is restricted ONLY to crack pixels.
    Non-crack pixels act as Dirichlet boundary conditions.

    Implements Equation (5) from arXiv 2602.12742 / Equation (17) from Giakoumis 2006.

    Args:
        image      : input BGR image
        crack_mask : binary mask (255 = crack pixel)
        n_iter     : number of diffusion iterations (default 20)
        lambda_    : time step, must be <= 0.25 for stability
        K          : gradient sensitivity constant

    Returns:
        restored   : BGR image with cracks filled
    """
    assert lambda_ <= 0.25, "lambda must be <= 0.25 for numerical stability"

    restored = image.copy().astype(np.float32)
    crack_pixels = (crack_mask > 0)

    for c in range(3):
        I = restored[:, :, c].copy()

        for _ in range(n_iter):
            # Compute directional differences (N, S, E, W)
            DN = np.roll(I, -1, axis=0) - I   # North
            DS = np.roll(I,  1, axis=0) - I   # South
            DE = np.roll(I,  1, axis=1) - I   # East
            DW = np.roll(I, -1, axis=1) - I   # West

            # Perona-Malik conductivity function: g(∇I) = 1 / (1 + (|∇I|/K)^2)
            cN = 1.0 / (1.0 + (np.abs(DN) / K) ** 2)
            cS = 1.0 / (1.0 + (np.abs(DS) / K) ** 2)
            cE = 1.0 / (1.0 + (np.abs(DE) / K) ** 2)
            cW = 1.0 / (1.0 + (np.abs(DW) / K) ** 2)

            # Discrete update: Equation (17)
            I_new = I + lambda_ * (cN * DN + cS * DS + cE * DE + cW * DW)

            # CRITICAL: Only update crack pixels — non-crack pixels are fixed (Dirichlet BC)
            I[crack_pixels] = I_new[crack_pixels]

        restored[:, :, c] = I

    return np.clip(restored, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def fill_cracks(image_path, crack_mask, method="AD", output_dir="results", window_size=5):
    """
    Fill cracks in a painting using either MTM or AD method.

    Args:
        image_path  : path to original painting
        crack_mask  : binary crack mask from top_hat_detection.py
        method      : "MTM" or "AD" (Anisotropic Diffusion)
        output_dir  : folder to save results
        window_size : only for MTM

    Returns:
        restored    : restored BGR image
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load: {image_path}")

    print(f"[Filling] Method: {method}")

    if method == "MTM":
        restored = mtm_filter(image, crack_mask, window_size=window_size)
    elif method == "AD":
        restored = anisotropic_diffusion(image, crack_mask)
    else:
        raise ValueError("method must be 'MTM' or 'AD'")

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_restored_{method}.png")
    cv2.imwrite(out_path, restored)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original (with cracks)")
    axes[0].axis("off")

    axes[1].imshow(crack_mask, cmap="gray")
    axes[1].set_title("Crack Mask")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Restored ({method})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_filling_{method}.png"), dpi=150)
    plt.close()

    print(f"[Filling] Saved: {out_path}")
    return restored


if __name__ == "__main__":
    import sys
    from top_hat_detection import detect_cracks

    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_painting.jpg"
    method     = sys.argv[2] if len(sys.argv) > 2 else "AD"
    threshold  = int(sys.argv[3]) if len(sys.argv) > 3 else 23

    print(f"Running crack filling on: {image_path}")
    crack_mask, _ = detect_cracks(image_path, threshold=threshold)
    restored = fill_cracks(image_path, crack_mask, method=method)
    print("Done.")
