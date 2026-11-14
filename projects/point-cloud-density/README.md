# Impact of Point Cloud Density on 3D Object Classification

How few points does a 3D point cloud need before classification accuracy collapses? A controlled benchmark of **3 samplers × 3 classifiers × 2 datasets × 3 seeds**, trained and tested at each density.

## Problem

The cost of many point cloud methods grows with the number of points, but there are no clear guidelines on how much to reduce. Most studies use clean data (ModelNet40) and train at high density, then test sparse clouds.

## Setup

| | |
|---|---|
| **Datasets** | ModelNet40 (CAD, 40 classes, 12,308 clouds) and ScanObjectNN OBJ_BG (real scans, 15 classes, 2,890 clouds) |
| **Densities** | ScanObjectNN: 1024 → 64 points · ModelNet40: 512 → 64 points (from 2048) |
| **Sampling** | Random, Farthest Point Sampling (FPS), voxel downsampling |
| **Models** | PointNet++, PointNet++ with density-scaled radii, DGCNN, Triangle-Net (normals re-estimated after sampling) |
| **Protocol** | Train from scratch at each density (30 epochs, Adam, cosine LR), stratified 70/15/15 split per seed (42/43/44), index hashes checked to rule out leakage |
| **Statistics** | Paired t-test with Holm correction; accuracy, macro-F1, training time |
| **Hardware** | Kaggle, Tesla T4, PyTorch 2.11, deterministic cuDNN |

## Key results

- **Smallest density without relevant loss:** **512 points on ScanObjectNN** (loss ≤ 2.8 p.p., and DGCNN trains **58% faster**) and **256 points on ModelNet40** (loss ≤ 2.3 p.p.). That is half of the largest density tested in each dataset.
- **At 64 points:** PointNet++ falls to 36.9-59.4% on ScanObjectNN and 83.8-85.9% on ModelNet40.
- **Per-class analysis:** the loss is spread across many classes; models drift toward predicting the larger classes, so **rare classes suffer most**.
- **Sampling:** random sampling often did best, but it redraws points every epoch (acting as data augmentation), so this conclusion holds only for this protocol.
- **Training cost:** DGCNN drops from 19.4 to 2.5 min (1024 → 64 points); Triangle-Net barely changes.

## Limitations (stated in the paper)

Only 3 seeds per setting, a single training configuration, a small ScanObjectNN test set (434 clouds), approximate inference/memory measurements, and a 3 p.p. threshold fixed after exploratory analysis.

## Reproduce

Full code and notebooks: [larissa-acosta/point-cloud-density-classification](https://github.com/larissa-acosta/point-cloud-density-classification). The experiments were run on Kaggle (one notebook per sampling method, identical except for the sampler).

```bash
pip install -r requirements.txt   # PyTorch 2.x, CUDA
```

## My contribution

I wrote the code for all experiments, curated the datasets, and wrote the paper.

## Authors

Arthur Capellazzi Fontana Amaral, Ayumi Sato Brizola, Gabriel Berto Beckauser, Larissa Cristina Alves da Costa · Projeto Transformador II, PUCPR

📄 Paper: [`paper.pdf`](paper.pdf)
