# HPA Protein Localization — Multi-label Classification

Multi-label classification of 28 subcellular protein localization compartments from 4-channel fluorescence microscopy images (red/green/blue/yellow channels), using the Human Protein Atlas dataset (31,072 training images).

## Scripts

### `hpa/train_hpa.py`
Training script for a 4-channel CNN (EfficientNet/ResNet via timm). Reads `train.csv` and the `train/` image folder. Logs per-epoch metrics to `logs/train_log.jsonl` and checkpoints to `logs/checkpoints/`. Supports live learning-rate override via `logs/live_config.json` between epochs without restarting.

### `hpa/agent_hpa.py`
Autonomous training agent that reads all historical run results from `logs/runs/` and reasons about which hyperparameter configuration to try next — no fixed grid search. Diagnoses overfitting, convergence speed, and per-class weaknesses to make evidence-based decisions. Launches `train_hpa.py` as a subprocess and archives completed runs automatically.

### `hpa/dashboard_hpa.py`
Flask web dashboard (port 8768) for monitoring training in real time. Provides dataset exploration (class distribution, co-localization, channel stats, image browser), live training metrics (loss/F1 curves, per-class F1 heatmap), agent reasoning log, post-analysis with biological compartment groupings and atlas annotation readiness assessment, and a chatbot interface for querying results.

## Data

### `hpa/train.csv`
Label file from the Kaggle HPA competition — 31,072 rows with `Id` and `Target` (space-separated class indices, 0–27).

### `train/` image folder (not included)
Download from Kaggle: [Human Protein Atlas Image Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification/data). Each sample has 4 PNG files: `{Id}_red.png`, `{Id}_green.png`, `{Id}_blue.png`, `{Id}_yellow.png`.

## Classes (28 subcellular compartments)

| # | Compartment | # | Compartment |
|---|-------------|---|-------------|
| 0 | Nucleoplasm | 14 | Actin filaments |
| 1 | Nuclear membrane | 15 | Focal adhesion sites |
| 2 | Nucleoli | 16 | Microtubule organizing center |
| 3 | Nucleoli fibrillar center | 17 | Centrosome |
| 4 | Nuclear speckles | 18 | Mitotic spindle |
| 5 | Nuclear bodies | 19 | Midbody |
| 6 | Endoplasmic reticulum | 20 | Lysosomes |
| 7 | Golgi apparatus | 21 | Plasma membrane |
| 8 | Peroxisomes | 22 | Cell junctions |
| 9 | Endosomes | 23 | Cytosol |
| 10 | Lysosomes | 24 | Intermediate filaments |
| 11 | Microtubules | 25 | Aggresome |
| 12 | Mitotic spindle | 26 | Cytoplasmic bodies |
| 13 | Cytoskeleton | 27 | Rods & rings |

## Dependencies

```
torch
timm
flask
numpy
pandas
Pillow
scikit-learn  # optional, falls back to numpy split
```

## Usage

```bash
# 1. Run the autonomous agent (launches training automatically)
python hpa/agent_hpa.py

# 2. Or run training directly with a config file
# Create logs/train_config.json first, then:
python hpa/train_hpa.py

# 3. Start the dashboard
python hpa/dashboard_hpa.py
# Open: http://localhost:8768
```

### Example `logs/train_config.json`
```json
{
  "model": "efficientnet_b0",
  "lr": 1e-4,
  "batch_size": 32,
  "epochs": 30,
  "img_size": 224,
  "pretrained": true,
  "val_split": 0.1,
  "num_workers": 4
}
```

```bibtex
@software{nguyen_2026_hpa,
  author    = {Nguyen, Huu Phong and Chen, Jake},
  title     = {HPA Dashboard: A Reusable Imaging Workflow Package with an Auto-Research Training Agent},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1},
  doi       = {10.5281/zenodo.20801879},
  url       = {https://doi.org/10.5281/zenodo.20801879}
}
```
