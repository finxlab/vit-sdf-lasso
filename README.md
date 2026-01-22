# From vision to value: Stock chart image-driven factors and their pricing power

<p align="center">
  <img src="figure/framework.png" alt="Architecture" width="600">
</p>

## Overview

A deep learning framework for constructing stock chart image–driven factors and testing their pricing power in asset markets.

## Project Structure
```
vit-sdf-lasso/
├── image_generation/          # Stock chart image generation
│   ├── data_preprocess.py     # CRSP data preprocessing
│   ├── gray_image_cnn_xiu.py  # Grayscale images for CNN (Jiang et al., 2023)
│   └── rgb_image_vit_byun.py  # RGB images for ViT (Byun et al., 2025)
│
├── models/                    # Pre-trained models and inference
│   ├── weights/               # Model weights (5 seeds each)
│   │   ├── cnn/
│   │   └── vit/
│   └── inference.py           # Generate predictions from images
│
├── sdf_analysis/              # SDF analysis pipeline
│   ├── data_preprocess.py     # Prepare CRSP data for portfolio construction
│   ├── sorting_portfolio.py   # Construct 3×3 size-signal sorted portfolios
│   ├── portfolio_performance.ipynb  # Performance metrics (Sharpe, MDD, etc.)
│   ├── univariate_beta_scaling.py   # Compute factor betas for penalty weights
│   ├── double_lasso_selection.py    # Double-selection LASSO with CV
│   ├── sdf_loading.ipynb      # Estimate SDF loadings (post-LASSO OLS)
│   ├── visualize.ipynb        # t-SNE and heatmap figures
│   └── appendix_visualize.ipynb     # Robustness check figures
│
├── figure/                    
├── requirements.txt
└── README.md
```

## Data
```
Public Data (included in `sdf_analysis/data/`)

| Data | Source |
|------|--------|
| Test portfolios | [global-q.org](https://global-q.org) |
| Q5 factors | [global-q.org](https://global-q.org) |
| FF3, FF5 factors | [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) |
| Global Factor Data | [jkpfactors.com](https://jkpfactors.com/) |
```

## Usage

1. Image Generation

Generate stock chart images from CRSP daily data.
```bash
cd image_generation
python data_preprocess.py        # Preprocess CRSP → data/stock/
python rgb_image_vit_byun.py     # Generate RGB images for ViT
python gray_image_cnn_xiu.py     # Generate grayscale images for CNN
```

2. Model Inference

Generate return predictions using pre-trained models.
```bash
cd models
# Edit MODEL_TYPE and IMAGE_DIR in inference.py
python inference.py              # Output: pred/vit/, pred/cnn/
```

3. SDF Analysis
```bash
cd sdf_analysis

# Step 1: Prepare CRSP data
python data_preprocess.py        # Output: data/*.pkl

# Step 2: Construct sorted portfolios
python sorting_portfolio.py      # Output: sorted_portfolio/*.csv

# Step 3: Compute factor returns and performance metrics
# Run: portfolio_performance.ipynb
# Output: factor_port/*.csv, figures

# Step 4: Compute univariate betas for LASSO penalty
python univariate_beta_scaling.py   # Output: data/beta_k.csv

# Step 5: Run double-selection LASSO (200 seeds, parallelized)
python double_selection_lasso.py    # Output: result/*/lasso*.csv

# Step 6: Estimate SDF loadings
# Run: sdf_loading.ipynb
# Output: result/*_t_stat_v2.csv

# Step 7: Generate figures
# Run: visualize.ipynb, appendix_visualize.ipynb
```

Note: Data will be made available upon request. Please contact jybyun@hanyang.ac.kr for your request.                                         
