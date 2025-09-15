# Replicating Guess-and-Learn Results

This document explains how to reproduce the full experiment grid and how to regenerate any single experiment. It assumes you are running from the repository root.

> **Runners & outputs**
> - Runner: `src/run_all.py`
> - Outputs: JSON + PNG written under the directory you pass via `--output_dir` (default: `./runs/`)
> - Tracks: `G&L-SO`, `G&L-PO`, `G&L-SB_<K>`, `G&L-PB_<K>`
> - Devices: default **CPU**; pass `--devices cuda` for GPU
> - Seeds: pass one or many, e.g. `--seeds 0 1 2 3 4`

---

## 0) Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# (Optional) pre-download/cache datasets & weights for offline runs
python src/scripts/prepare_cache.py
```

---

## 1) Full reproduction (scripted)

Run the full grid (as used for the repository results) with the provided script:

```bash
chmod +x scripts/reproduce_all.sh
./scripts/reproduce_all.sh
```

Notes:
- The script enumerates datasets (e.g., **MNIST, Fashion-MNIST, CIFAR10, SVHN, AG News**), models (`cnn`, `resnet50`, `vit-b-16`, `bert-base`, `knn`, `perceptron`, `text-knn`, `text-perceptron`), strategies (`random`, `confidence`, `least_confidence`, `margin`, `entropy`), tracks, seeds, and subsets.
- Override defaults with environment variables:
  ```bash
  OUTDIR=./runs_big DEVICES=cuda ./scripts/reproduce_all.sh
  ```

---

## 2) Regenerating a specific experiment

Filename pattern:
```
<dataset>_<model>_<strategy>_<track>[_K]_seed<SEED>[_s<SUBSET>][_reset]_{results.json|plot.png}
```

Recreate any single run like:

```bash
# Example: MNIST / CNN / entropy / online (SO), seed 3, subset 300
python src/run_all.py \
  --datasets mnist \
  --models cnn \
  --strategies entropy \
  --tracks 'G&L-SO' \
  --seeds 3 \
  --subset 300 \
  --output_dir ./runs
```

> **Tracks (glossary)**
> - `G&L-SO` — scratch, online
> - `G&L-PO` — pretrained, online
> - `G&L-SB_<K>` — scratch, batch (budget K)
> - `G&L-PB_<K>` — pretrained, batch (budget K)
> Some runners also support `--reset` to re‑initialise models between batches (only use when your target filename contains `_reset`).

---

## 3) Artifacts & layout

After runs finish you should see files like:

```
runs/
  mnist_cnn_entropy_G&L-SO_seed0_s300_results.json
  mnist_cnn_entropy_G&L-SO_seed0_s300_plot.png
  ...
```

---

## 4) Determinism & performance notes

- **Seeds**: pass multiple via `--seeds 0 1 2 3 4` to reproduce mean/uncertainty from your full runs.
- **Subset**: `--subset 300` speeds up local testing; omit it to run on the full dataset (as in your release results).
- **Devices**: pass `--devices cuda` for GPU; CPU is the default and sufficient for basic checks.
- **Caching**: `src/scripts/prepare_cache.py` pre‑downloads datasets/weights.

---

## 5) Cite

If you use this repository or the benchmark, please cite:

- Preprint: **Arnold, R. W. (2025)**, *Guess-and-Learn: A Benchmark for Cold-Start Adaptation*, DOI: **10.48550/arXiv.2508.21270**
- Software: see the “Cite this repository” panel (from `CITATION.cff`) on the GitHub sidebar.
- ORCID: https://orcid.org/0009-0001-0374-4692

---

## 6) Troubleshooting

- **Invalid combinations**: `run_all.py` validates combos; ensure pretrained models (`resnet50`, `vit-b-16`, `bert-base`) are paired with `G&L-PO`/`G&L-PB_<K>`.
- **Long runtimes**: run a single dataset/model/strategy/track first; use the full runs only for final figures/tables.
- **Plots not created**: confirm write permissions under `--output_dir`; ensure a non‑interactive matplotlib backend is available (CI uses `MPLBACKEND=Agg`).

