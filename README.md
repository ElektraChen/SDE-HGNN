# SDE-Driven Spatio-Temporal Hypergraph Neural Networks for Irregular Longitudinal fMRI Connectome Modeling in Alzheimer's Disease

**AMIA 2026 (Anonymous Submission)**

This repository provides the implementation for the paper *SDE-Driven Spatio-Temporal Hypergraph Neural Networks for Irregular Longitudinal fMRI Connectome Modeling in Alzheimer's Disease.*

Code + config only; **no preprocessed data** in the repo. 

---

## Overview

- **Data:** `data.py`, `data_oasis.py`, and `data_util.py` — cohort **I/O**, **signal reconstruction / preprocessing** (SDE-based where used), and **multi-visit tensors + labels** for training.
- **Backbone:** KNN **hypergraph** per time point → **HGNN** message passing; **SDE** models evolution across visits (default: EvolveHGNN + SDE).
- **Training:** Stratified **5-fold** cross-validation; metrics such as AUC, accuracy, sensitivity, specificity (see logs under `results/` after you run the code).

---

## Model variants (code)

| Name | Role |
|------|------|
| **EvolveHGNN + SDE** | Main: hypergraph conv + SDE over time (`sgcn_progress_sde_hgnn.py`, `evolveHGNN.py`). |
| **HGNN + latent SDE** | Static HGNN per visit + sequence SDE (`sgcn_progress_evolve_sde_hgnn.py`, `HGNN_model/`). |
| **Sparsity variant** | Subset node / hyperedge sparsity (`sgcn_progress_sde_hgnn_sparsity.py`). |
| **Graph + ODE** | Graph-ODE modules (`graphode.py`, `graphode_nopos.py`). |
| **GCN baselines** | Same training script can switch to GCN paths (`sgcn_progress_sde_gcn.py`, `sgcn_progress_v2.py`). |

---

## Train / validation / test split

**5-fold CV** (`kernel/train_eval_sgcn_postsde_progress_v2.py` → `k_fold`): each fold, **StratifiedKFold** holds out **1/5 ≈ 20%** as **test**. On the other **80%**, **train_test_split(test_size=0.25)** gives **25% of that block as val** and **75% as train** — i.e. about **60% train / 20% val / 20% test** of the full cohort per fold.

---

## Requirements

- Python 3.8+; GPU recommended.  
- Install dependencies from **`config/`** (e.g. `pip install -r config/requirements-*.txt`).  
- Typical stack: PyTorch, PyG, `torchsde`, `torchdiffeq`, scikit-learn, numpy/scipy/pandas.

---

## How to run

```bash
cd sdehgnn
# Point data roots inside data_oasis / data as on your machine
python main.py --isOASIS   # example; see main.py for --fold, --epochs, HGNN flags, etc.
```

Main entry: **`main.py`** → **`kernel/train_eval_sgcn_postsde_progress_v2.py`**.

---

## Project layout

```
sdehgnn/
├── README.md
├── main.py
├── data.py, data_oasis.py, data_util.py   # I/O, reconstruction, preprocess
├── utils.py, utils_graph.py, hyper_utils_torch.py, Imbalanced.py
├── config/              # environment lock files
├── lib/
└── kernel/              # models + training loop
```

---

## Acknowledgements

Part of the **HGNN convolution** follows **Feng et al., *Hypergraph Neural Networks* (AAAI 2019)**. **EvolveHGNN** and **SDE** parts are our extensions. Please cite the HGNN paper if you reuse those layers.

---

## License

See the repository license file if provided.

