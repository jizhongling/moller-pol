# Moller Polarimeter DAQ Analysis & Clustering

## Goals
- Extract peak-based features from raw FADC waveforms with ROOT.
- Cluster events using PCA/UMAP and neural autoencoders/VAEs plus KMeans/DBSCAN/HDBSCAN.
- Visualize clustered waveforms, summed spectra, and feature distributions.

## Components
- Feature extraction (ROOT C++): [AnaWaveform.C](AnaWaveform.C) writes [data/training-*.root](data) with per-peak time/peak/fwhm/area and timing differences between channels.
- Clustering and dimensionality reduction (Python): [main.py](main.py) loads ROOT features or full waveforms (for AE/VAE), applies normalization/PCA/UMAP/AE/VAE, and clusters via KMeans/DBSCAN/HDBSCAN; outputs labels to [data/labels-type{0|1}-method*.txt](data).
- Waveform visualization (ROOT C++): [DrawWaveform.C](DrawWaveform.C) overlays labels on waveforms and exports PDFs plus summed spectra.
- Feature distributions (ROOT C++): [DrawDistributions.C](DrawDistributions.C) plots histograms for time/area/fwhm branches.

## Data Flow
1. Raw data: Root files in [Rootfiles](Rootfiles) (tree `waveform` under `/mode_10_data/slot_3`).
2. Extraction: [AnaWaveform.C](AnaWaveform.C) → [data/training-*.root](data) (tree `T`).
3. Clustering: [main.py](main.py) → [data/labels-type{0|1}-method*.txt](data).
4. Visualization: [DrawWaveform.C](DrawWaveform.C) → [plots](plots) PDFs and spectra.

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

### Neural dimensionality reduction
- Train AE + cluster:
```shell
python main.py --type 0 --method 2 --eps 15 --autoencoder --latent-dim 10 --epochs 300 --save-model
```
- Load saved AE/VAE: add `--load-autoencoder models/autoencoder-type0-dim10.pt` or `--load-vae models/vae-type0-dim10.pt`.
- Train VAE instead of AE: replace `--autoencoder` with `--vae`.

## Common Flags (partial)
- `--type {0|1}`: full dataset vs. recluster a previous label.
- `--label N`: label to recluster (type 1).
- `--norm`: apply StandardScaler.
- `--pca` `--pca-components N`; `--umap` `--umap-components N`.
- `--autoencoder` / `--vae`, `--latent-dim N`, `--epochs`, `--batch-size`, `--learning-rate`.
- `--method {0|1|2}`: KMeans / DBSCAN / HDBSCAN.
- `--save-model`, `--model-dir DIR`.

## Outputs
- Labels: [data/labels-type{0|1}-method*.txt](data).
- Models: [models](models) (e.g., `autoencoder-type{type}-dim{d}.pt`, `vae-type{type}-dim{d}.pt`).
- Plots: [plots](plots) PDFs such as Waveform-run*-method*.pdf, spectra, distributions.
