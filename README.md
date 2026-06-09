# 🐟 MLP Activation Comparison Toolbox for Pelagic Fish Habitat Mapping
### *A QGIS Processing Plugin for FMA 573 — Indian Ocean, Indonesia*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![QGIS](https://img.shields.io/badge/QGIS-3.x-589632?style=flat-square&logo=qgis&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-MLP-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Research-blue?style=flat-square)

**Part of a Master's thesis research — Universitas Gadjah Mada, 2025**

[Overview](#-overview) · [Results](#-comparative-results) · [Installation](#-installation) · [Usage](#-usage) · [Data Requirements](#-data-requirements) · [Citation](#-citation)

</div>

---

## 📌 Overview

This repository contains **three QGIS Custom Processing Toolbox scripts** for predicting pelagic fish habitat probability using Multi-Layer Perceptron (MLP) deep learning. Each script is identical in architecture except for the **activation function**, enabling a direct empirical comparison on multi-parameter oceanographic satellite data.

The toolbox was developed as part of a comparative study investigating which activation function best captures the non-linear, multivariate nature of oceanographic conditions in **Fisheries Management Area (FMA) 573** — the Indian Ocean south of Java, Bali, and Nusa Tenggara, Indonesia.

```
Research Question:
Which activation function — tanh, ReLU, or Sigmoid — is optimal
for MLP-based pelagic fish habitat modeling in FMA 573?
```

---

## 📊 Comparative Results

> Tested on actual catch logbook data from Cilacap Fishing Port (2025) with 7 satellite-derived oceanographic variables.

| Metric | tanh ⭐ | ReLU | Sigmoid |
|---|:---:|:---:|:---:|
| Training Accuracy | **85.57%** | 83.74% | 75.00% |
| CV Accuracy (5-Fold) | **83.95%** | 82.12% | 74.20% |
| CV Std | 0.0× | 0.0341 | 0.0299 |
| F1-Score (weighted) | **0.85** | 0.83 | 0.73 |
| ROC-AUC | 0.8343 | **0.8355** | 0.7545 |
| AUC Interpretation | Good | Good | Fair |
| Execution Time | ~50s | **7.5s** | ~49s |

### 🔑 Key Findings

- **tanh** achieves the best overall classification balance and F1-score — recommended for production use
- **ReLU** is competitive in ROC-AUC with significantly faster execution (~7× faster)
- **Sigmoid** underperforms on all metrics; near-exclusive dependence on SST in Permutation Importance suggests it fails to capture multivariate oceanographic interactions
- **SST** is the dominant predictor across all three models, confirming its role as the primary driver of pelagic habitat distribution in FMA 573

### 📈 Permutation Importance (variable ranking)

| Rank | Variable | tanh | ReLU | Sigmoid |
|:---:|---|:---:|:---:|:---:|
| 1 | Sea Surface Temperature (SST) | 0.1386 | 0.1860 | 0.1764 |
| 2 | Salinity | 0.0976 | 0.0907 | 0.0134 |
| 3 | Current Velocity | 0.0866 | 0.0433 | 0.0000 |
| 4 | Bathymetry | 0.0762 | 0.0537 | 0.0000 |
| 5 | Chlorophyll-a | 0.0681 | 0.0081 | 0.0000 |
| 6 | Sea Surface Height (SSH) | 0.0400 | 0.0443 | 0.0234 |
| 7 | Coral Reef | 0.0000 | 0.0000 | 0.0000 |

---

## 📁 Repository Structure

```
mlp-fish-habitat-fma573/
│
├── mlp_tanh.py          # Model A — tanh activation (baseline, recommended)
├── mlp_relu.py          # Model B — ReLU activation
├── mlp_sigmoid.py       # Model C — Sigmoid (logistic) activation
│
├── README.md            # This file
└── LICENSE              # MIT License
```

---

## ⚙️ Installation

### Prerequisites

| Software | Version | Notes |
|---|---|---|
| QGIS | ≥ 3.x | With Processing Framework enabled |
| Python | ≥ 3.x | Bundled with QGIS |
| scikit-learn | ≥ 1.0 | Via OSGeo4W or pip |
| numpy | any | Bundled with QGIS |
| pandas | any | Via OSGeo4W or pip |
| matplotlib | any | For ROC curve export |

### Install missing Python libraries in QGIS

Open **OSGeo4W Shell** (Windows) or terminal and run:

```bash
pip install scikit-learn pandas matplotlib
```

Or via QGIS Python Console:

```python
import subprocess
subprocess.run(['pip', 'install', 'scikit-learn', 'pandas', 'matplotlib'])
```

### Add scripts to QGIS Processing Toolbox

1. Open QGIS → **Processing** menu → **Toolbox**
2. Click the ⚙️ icon → **Add Script to Toolbox**
3. Select all three `.py` files
4. Scripts will appear under **Perikanan** group in the Toolbox

---

## 🚀 Usage

### Step 1 — Prepare your data

| Input | Format | Description |
|---|---|---|
| Fishing Points | Vector (Point) | Catch coordinates from vessel logbook / VMS |
| SST | Raster (.tif) | Sea Surface Temperature — MODIS via GEE |
| Chlorophyll-a | Raster (.tif) | CHL-a — MODIS via GEE |
| Bathymetry | Raster (.tif) | ETOPO — via GEE |
| Coral Reef | Raster (.tif) | Allen Coral Atlas — via GEE |
| Salinity | Raster (.tif) | HYCOM — via GEE |
| SSH | Raster (.tif) | Sea Surface Height — HYCOM/Altimetry via GEE |
| Current Velocity | Raster (.tif) | HYCOM — via GEE |

> **Note:** All rasters should cover the same spatial extent and be in WGS 84 (EPSG:4326). The script handles resampling internally using nearest neighbor interpolation.

### Step 2 — Run the model

1. Double-click the desired script in the Processing Toolbox
2. Assign each input layer to the corresponding parameter
3. Set the output path for the prediction raster
4. Click **Run**

### Step 3 — Read the outputs

The script generates two outputs:

**① Prediction raster** (`.tif`) — pixel-by-pixel habitat probability map
- Values range from `0.0` (low suitability) to `1.0` (high suitability)
- Apply a gradient color ramp in QGIS for visualization

**② ROC curve image** (`.png`) — saved automatically to the same folder as the output raster
- Named `roc_curve_tanh.png` / `roc_curve_relu.png` / `roc_curve_logistic.png`

### Step 4 — Interpret the log

The Processing Log displays:

```
CV Accuracy (5-Fold) mean : 0.8395
CV Accuracy (5-Fold) std  : 0.02xx
Accuracy          : 0.8557
F1-Score (weighted): 0.85
ROC-AUC Score     : 0.8343
Interpretasi AUC  : Good (0.80 - 0.89)

PERMUTATION IMPORTANCE
  SST                 : 0.1386 (+/- 0.0xxx)
  Salinity            : 0.0976 (+/- 0.0xxx)
  ...
```

---

## 📡 Data Requirements & Sources

All satellite data can be accessed freely via **Google Earth Engine (GEE)**. Sample GEE code snippets:

```javascript
// SST — MODIS Terra
var sst = ee.ImageCollection('MODIS/061/MOD11A1')
  .filterDate('2025-01-01', '2025-12-31')
  .select('LST_Day_1km')
  .mean();

// Chlorophyll-a — MODIS Aqua
var chl = ee.ImageCollection('NASA/OCEANDATA/MODIS-Aqua/L3SMI')
  .filterDate('2025-01-01', '2025-12-31')
  .select('chlor_a')
  .mean();

// Define FMA 573 region
var fma573 = ee.Geometry.Rectangle([102, -15, 130, -6]);
```

---

## 🏗️ Model Architecture

All three scripts share identical architecture — only `activation` differs:

```python
MLPClassifier(
    hidden_layer_sizes = (3,),        # 1 hidden layer, 3 neurons
    activation         = 'tanh',      # 'tanh' | 'relu' | 'logistic'
    solver             = 'lbfgs',     # L-BFGS optimizer
    alpha              = 5.0,         # L2 regularization (strong)
    max_iter           = 1500,        # maximum iterations
    random_state       = 42           # reproducibility seed
)
```

**Pre-processing pipeline (identical for all models):**

```
Raw coordinates → IQR outlier removal → Gaussian noise augmentation (σ=0.12)
→ Pseudo-absence generation (±1.0 SD) → RobustScaler normalization
→ 5-Fold Cross Validation → MLP Training
```

---

## 📋 Evaluation Metrics

Each script reports the following in the Processing Log:

- ✅ **5-Fold Cross-Validation** — mean accuracy ± std
- ✅ **Accuracy, Precision, Recall, F1-Score** — per class and weighted
- ✅ **ROC-AUC Score** — with automated interpretation (Excellent/Good/Fair/Poor)
- ✅ **ROC Curve** — exported as PNG
- ✅ **Permutation Importance** — variable contribution ranking (XAI)

---

## 🔬 Research Context

| Item | Detail |
|---|---|
| **Title** | A Comparative Study of Tanh, ReLU, and Sigmoid for Optimizing MLP in Pelagic Fish Habitat Mapping |
| **Study Area** | FMA 573 — Indian Ocean south of Java, Bali, Nusa Tenggara |
| **Institution** | Master's Program in Remote Sensing, Faculty of Geography, UGM |
| **Author** | Rosyid Paundra Gamawan (NIM: 25/562471/PGE/01722) |
| **Year** | 2025 |
| **Ground Truth** | Catch logbook data — Cilacap Fishing Port (Pelabuhan Perikanan Samudera Cilacap) |

---

## 📖 Citation

If you use this toolbox in your research, please cite:

```bibtex
@misc{gamawan2025mlpfma573,
  author    = {Gamawan, Rosyid Paundra},
  title     = {MLP Activation Comparison Toolbox for Pelagic Fish Habitat Mapping — FMA 573},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/rosyidpaundra/mlp-fish-habitat-fma573},
  note      = {Master's Thesis, Remote Sensing Program, Universitas Gadjah Mada}
}
```

**Related paper (conference proceeding):**

> Gamawan, R. P. (2025). A Comparative Study of Tanh, ReLU, and Sigmoid for Optimizing Multi-Layer Perceptron in Pelagic Fish Habitat Mapping (Study Area: FMA 573). *Conference Proceedings.*

---

## 🤝 Contributing

Contributions are welcome, especially:
- Testing on different FMA regions (571, 572, 574, etc.)
- Adding additional activation functions (Leaky ReLU, ELU)
- Integration with additional satellite data sources
- GUI improvements for the QGIS plugin

Please open an **Issue** or submit a **Pull Request**.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Rosyid Paundra Gamawan**
Master's Program in Remote Sensing — Faculty of Geography, Universitas Gadjah Mada
Yogyakarta, Indonesia

[![GitHub](https://img.shields.io/badge/GitHub-rosyidpaundra-181717?style=flat-square&logo=github)](https://github.com/rosyidpaundra)

---

<div align="center">
<sub>Built with ❤️ for sustainable fisheries management in Indonesia</sub><br>
<sub>Universitas Gadjah Mada · Faculty of Geography · Remote Sensing Program · 2025</sub>
</div>
