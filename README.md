
# Crack Detection and Removal in Digitized Paintings

**ECL-415 Digital Image Processing | IIITN | Group Project**

A comparative implementation of **classical, modern, and experimental approaches** for crack detection and virtual restoration of digitized paintings.

---

## Reference Papers

- **[Classical]** Giakoumis et al., *"Digital Image Processing Techniques for the Detection and Removal of Cracks in Digitized Paintings"*, IEEE Transactions on Image Processing, Vol. 15, No. 1, 2006.
- **[Modern]** Cuch-Guillén et al., *"Synthetic Craquelure Generation for Unsupervised Painting Restoration"*, arXiv:2602.12742, February 2026.

---

## Pipeline Overview

### Classical Pipeline (Giakoumis 2006)


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

### Modern Pipeline (arXiv 2026 Inspired)
```
Input Painting
     │
     ▼
[1] Top-Hat Transform (Classical Base)
     │
     ▼
[2] Morphological + Edge-Aware Refinement
     │
     ▼
[3] SegFormer Zero-Shot Guidance (Weak Texture Prior)
     │
     ▼
[4] Crack Filling (AD / MTM)
     ├── Method A: Modified Trimmed Mean (MTM) Filter
     └── Method B: Anisotropic Diffusion (AD)
     │
     ▼
Improved Restoration (Robust to Texture Variations)
```

> SegFormer is used in **zero-shot mode** as a *soft refinement signal*, not as a fully trained crack detector.

---

## Additional Contributions (This Project)

### Synthetic Crack Generator (Bezier-based)

- Implements **Bezier curve-based craquelure generation** (from 2026 paper)
- Generates realistic:
  - crack masks
  - damaged paintings
- Used for:
  - qualitative validation
  - pipeline testing (not full training)

---

### Novel Method: Exemplar-Based Patch Inpainting (EBPI)

A texture-driven inpainting method:

- Instead of diffusion or averaging:
  - finds best matching patches from non-crack regions
  - copies texture into damaged areas
- Conceptually closer to modern inpainting systems

**Pipeline:**
``` Crack Mask → Patch Search → Best Match → Texture Transfer → Fill ```


**Performance (Sample Run):**

| Method | PSNR (dB) | SSIM | MSE | Time (s) |
|---|---|---|---|---|
| Classical AD | 30.87 | 0.9045 | 53.21 | 0.03 |
| Modern AD (Orientation-Sensitive) | 30.95 | 0.9061 | 52.22 | 0.13 |
| **Novel EBPI** | 30.28 | 0.8979 | 60.88 | 19.39 |

> EBPI preserves texture well but is computationally expensive and less stable than diffusion-based methods.

---

## Project Structure


```
crack-detection-painting-restoration/
├── main.py
├── compare.py # Classical vs Modern comparison
├── requirements.txt
├── README.md
│
├── classical/
│ ├── top_hat_detection.py
│ └── crack_filling.py
│
├── modern/
│ ├── modern_pipeline.py
│ ├── segformer_refinement.py
│ ├── synthetic_crack_generator.py
│ └── tgbi.py # EBPI implementation
│
├── utils/
│ └── metrics.py
│
├── data/
└── results/
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/ParthCh300X/crack-detection-painting-restoration.git
cd crack-detection-painting-restoration

# Install dependencies
pip install -r requirements.txt
```

# Painting Crack Detection & Restoration

A computer vision pipeline for detecting and restoring cracks in paintings, comparing classical image processing techniques with a modern deep learning approach.

---

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Results](#results)
- [Observations](#observations)
- [Team](#team)

---

## Overview

<!-- TODO: Add a 2–4 sentence description of the project. What problem does it solve? What techniques does it use at a high level? Example: "This project implements two pipelines for crack detection in artwork images — a classical image processing pipeline and a modern deep learning pipeline using SegFormer. Restoration is performed using Morphological Top-Hat (MTM) and Anisotropic Diffusion (AD) methods." -->

---

## Usage

### Prerequisites

<!-- TODO: Add setup/installation instructions. Example:
```bash
pip install -r requirements.txt
```
Also mention any model weights that need to be downloaded, or environment requirements (Python version, CUDA, etc.).
-->

### Classical Pipeline

```bash
python main.py
python main.py data/painting.jpg
```

### Modern Pipeline

```bash
python modern/modern_pipeline.py data/painting.jpg
```

### Comparison (Key Result)

```bash
python compare.py data/painting.jpg
```

### Synthetic Crack Generation

```bash
python modern/modern_pipeline.py data/painting.jpg --demo-synth
```

---

## Results

Outputs are saved in:

```
results/<image_name>/
```

Each run produces the following outputs:

- Crack masks
- Top-hat outputs
- Restored images (MTM / AD)
- Modern pipeline outputs
- Comparison figures

<!-- TODO: Optionally add sample result images here using:
![Sample Result](results/example/comparison.png)
-->

---

## Observations

- **Anisotropic Diffusion (AD)** consistently outperforms MTM for wider cracks.
- **Modern pipeline** improves robustness in highly textured regions.
- **SegFormer (zero-shot)** helps suppress false positives but is not sufficient alone.
- **EBPI** produces visually rich textures but is slower and less stable.

---

## Team

| Name | Roll No. |
|------|----------|
| Keshav Tak | BT23ECI039 |
| Kaushik Kumar | BT23ECI044 |
| Parth Chaudhary | BT23ECI045 |
| Shreyas Khare | BT23ECI058 |

**IIIT Nagpur — ECE (IoT), Semester VI**

---

