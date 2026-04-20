# Crack Detection and Removal in Digitized Paintings

**ECL-415 Digital Image Processing | IIITN | Group Project**

A comparative implementation of classical and learning-based approaches for crack detection and virtual restoration of digitized paintings.

---

## Reference Papers

- **[Classical]** Giakoumis et al., *"Digital Image Processing Techniques for the Detection and Removal of Cracks in Digitized Paintings"*, IEEE Transactions on Image Processing, Vol. 15, No. 1, 2006.
- **[Modern]** Cuch-Guillén et al., *"Synthetic Craquelure Generation for Unsupervised Painting Restoration"*, arXiv:2602.12742, February 2026.

---

## Pipeline Overview

```
Input Painting
     │
     ▼
[1] Top-Hat Transform (Morphological)
     │
     ▼
[2] Thresholding → Binary Crack Mask
     │
     ▼
[3] Size-based Noise Filtering
     │
     ▼
[4] Crack Filling
     ├── Method A: Modified Trimmed Mean (MTM) Filter
     └── Method B: Anisotropic Diffusion (AD)
     │
     ▼
Restored Painting + Metrics (PSNR / SSIM / MSE)
```

---

## Project Structure

```
crack-detection-painting-restoration/
├── main.py                        # Run full pipeline
├── requirements.txt
├── README.md
│
├── classical/
│   ├── top_hat_detection.py       # Morphological crack detection
│   └── crack_filling.py           # MTM filter + Anisotropic Diffusion
│
├── utils/
│   └── metrics.py                 # PSNR, SSIM, MSE, F1
│
├── data/                          # Input painting images (add your own)
│
└── results/                       # Output images (auto-generated)
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/crack-detection-painting-restoration.git
cd crack-detection-painting-restoration

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run on all images in data/ folder
```bash
python main.py
```

### Run on a single image
```bash
python main.py data/painting.jpg
```

### Run on a single image with custom threshold
```bash
python main.py data/painting.jpg 19
```

> **Threshold tuning:** Values between 15–30 work best. Lower threshold → more cracks detected. Higher → fewer but more precise.

### Run individual modules
```bash
# Detection only
python classical/top_hat_detection.py data/painting.jpg 23

# Filling only (after detection)
python classical/crack_filling.py data/painting.jpg AD 23
```

---

## Results

All outputs saved to `results/<image_name>/`:

| File | Description |
|---|---|
| `*_crack_mask.png` | Binary crack map |
| `*_tophat.png` | Top-hat transform output |
| `*_restored_MTM.png` | MTM filter result |
| `*_restored_AD.png` | Anisotropic Diffusion result |
| `*_detection_result.png` | Side-by-side detection visualization |
| `*_filling_MTM.png` | Side-by-side MTM filling |
| `*_filling_AD.png` | Side-by-side AD filling |

---

## Sample Metrics

| Method | PSNR (dB) | SSIM | MSE |
|---|---|---|---|
| MTM Filter | ~28–32 | ~0.85–0.92 | ~15–40 |
| Anisotropic Diffusion | ~30–35 | ~0.88–0.95 | ~10–25 |

> AD consistently outperforms MTM for wider cracks while MTM is faster for thin cracks.

---

## Team

| Name | Roll No. |
|---|---|
| Keshav Tak | BT23ECI039 |
| Kaushik Kumar | BT23ECI044 |
| Parth Chaudhary | BT23ECI045 |
| Shreyas Khare | BT23ECI058 |

IIITN — ECE-IoT, Semester VI

---

## License

For academic use only.
