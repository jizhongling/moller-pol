# Moller Polarimeter DAQ Analysis & Clustering

## Goals
- Extract peak-based features from raw FADC waveforms with ROOT.
- **Unsupervised**: Cluster events using PCA/UMAP and neural autoencoders/VAEs plus KMeans/DBSCAN/HDBSCAN.
- **Supervised**: Train tree-based classifiers (Decision Tree, Random Forest, Gradient Boosting) on production mode FADC features with labeled data.
- Visualize clustered/classified waveforms, summed spectra, and feature distributions.

## Components
- Feature extraction (ROOT C++): [AnaWaveform.C](AnaWaveform.C) writes [data/training-*.root](data) with:
  - Gaussian fit features: per-peak time/peak/fwhm/area and timing differences between channels
  - Production mode features: `pinteg`/`ptime`/`pped`/`ppeak` (4 peaks × 4 channels + 6 differential timing channels)
- Clustering and dimensionality reduction (Python): [main.py](main.py) loads ROOT features or full waveforms (for AE/VAE), applies normalization/PCA/UMAP/AE/VAE, and clusters via KMeans/DBSCAN/HDBSCAN; outputs labels to [data/labels-type{0|1}-method*.txt](data).
- **Supervised classification (Python)**: [main.py](main.py) with `--type 2` trains tree-based models (Decision Tree, Random Forest, Gradient Boosting) on production mode features with labeled data; outputs predictions to [data/predictions-type2-method*.txt](data).
- Waveform visualization (ROOT C++): [DrawWaveform.C](DrawWaveform.C) overlays labels/predictions on waveforms and exports PDFs plus summed spectra.
- Feature distributions (ROOT C++): [DrawDistributions.C](DrawDistributions.C) plots histograms for time/area/fwhm branches.

## Data Flow

### Unsupervised Clustering (Type 0/1)
1. Raw data: Root files in [Rootfiles](Rootfiles) (tree `waveform` under `/mode_10_data/slot_3`).
2. Extraction: [AnaWaveform.C](AnaWaveform.C) → [data/training-*.root](data) (tree `T`).
3. Clustering: [main.py](main.py) → [data/labels-type{0|1}-method*.txt](data).
4. Visualization: [DrawWaveform.C](DrawWaveform.C) → [plots](plots) PDFs and spectra.

### Supervised Classification (Type 2)
1. Raw data with production mode branches: [Rootfiles](Rootfiles) (includes `pinteg`, `ptime`, `pped`, `ppeak`).
2. Extraction: [AnaWaveform.C](AnaWaveform.C) → [data/training-*.root](data) with production features.
3. Training: [main.py](main.py) `--type 2` loads labeled data from previous clustering results → trains tree classifier → [data/predictions-type2-method*.txt](data).
4. Visualization: [DrawWaveform.C](DrawWaveform.C) with `start_type=2` → [plots](plots) with true/predicted labels.

## Setup
- Activate Python venv (example):

```shell
source /home/jzl/venvs/ml/bin/activate
```

- Dependencies: uproot, numpy, pandas, scikit-learn, umap-learn, torch, hdbscan, ROOT 6 runtime.

## Procedures

### 1) Generate training ROOT features
```shell
root -l -b -q 'AnaWaveform.C(0)'
```

### 2) Cluster (type 0, normalize + PCA + HDBSCAN)
```shell
python main.py --type 0 --method 2 --eps 1.0 --norm --umap
```

### 3) Optional: recluster a label (type 1, label N, KMeans)
```shell
python main.py --type 1 --label N --method 0 --kclus 3 --norm --umap
```

### 4) Visualize labeled waveforms
```shell
root -l -b -q 'DrawWaveform.C()'
```

### 5) Plot feature distributions
```shell
root -l -b -q 'DrawDistributions.C()'
```

### 6) Supervised classification with production mode features (type 2)
```shell
# Train histogram gradient boosting classifier on labeled events
python main.py --type 2 --method 3 --ntree 100 --lr 0.1 --test-size 0.2
```

### 7) Visualize predictions from supervised model
```shell
# Edit DrawWaveform.C: set start_type = 2, method[2] = 3
root -l -b -q 'DrawWaveform.C()'
```

### Neural dimensionality reduction
- Train AE + cluster:
```shell
python main.py --type 0 --method 2 --eps 15 --autoencoder --latent-dim 10 --epochs 300 --save-model
```
- Load saved AE/VAE: add `--load-autoencoder models/autoencoder-type0-dim10.pt` or `--load-vae models/vae-type0-dim10.pt`.
- Train VAE instead of AE: replace `--autoencoder` with `--vae`.

## Common Flags (partial)

### Workflow Selection
- `--type {0|1|2}`: 0=full clustering, 1=recluster a previous label, 2=supervised classification on production features.
- `--label N`: label to recluster (type 1 only).

### Dimensionality Reduction
- `--norm`: apply StandardScaler.
- `--pca` `--pca-components N`; `--umap` `--umap-components N`.
- `--autoencoder` / `--vae`, `--latent-dim N`, `--epochs`, `--batch-size`, `--learning-rate`.
- `--load-autoencoder PATH`, `--load-vae PATH`: use pre-trained models.

### Clustering Methods (Type 0/1)
- `--method {0|1|2}`: 0=KMeans, 1=DBSCAN, 2=HDBSCAN.
- `--kclus N`: number of clusters for KMeans.
- `--eps F`: epsilon for DBSCAN/HDBSCAN.

### Tree-Based Classification (Type 2)
- `--method {0|1|2|3}`: 0=DecisionTree, 1=RandomForest, 2=GradientBoosting, 3=HistGradientBoosting.
- `--ntree N`: number of trees/iterations (default: 100).
- `--lr F`: learning rate for gradient boosting (default: 0.1).
- `--test-size F`: fraction for test set (default: 0.2).

### General
- `--save-model`, `--model-dir DIR`.
- `--seed N`: random seed.

## Outputs
- **Clustering labels**: [data/labels-type{0|1}-method*.txt](data) (format: `event_id label`).
- **Classification predictions**: [data/predictions-type2-method*.txt](data) (format: `event_id true_label predicted_label`).
- **Models**: [models](models)
  - Neural: `autoencoder-type{type}-dim{d}.pt`, `vae-type{type}-dim{d}.pt`
  - Tree: `tree-type2-method{m}.pkl`, `scaler-type2-method{m}.pkl`
- **Plots**: [plots](plots) PDFs such as:
  - Clustering: `Waveform-run{N}-method{M0}-method{M1}.pdf`
  - Predictions: `Waveform-run{N}-predictions-method{M}.pdf`
  - Spectra and feature distributions.
