# Guess-and-Learn (G&L)
**Benchmarking early-phase adaptation by cumulative error**

G&L evaluates how quickly a model becomes useful in a **cold-start** setting. The learner
must predict a label for an unlabeled instance, receive the ground truth, update its
parameters, and repeat. The primary metric is the **cumulative number of mistakes**
over the sequence (the **error trajectory**). This exposes adaptation speed and the
effects of selection and training policies—information that final accuracy alone does not show.

---

## 1. Overview
- **Protocol:** Sequential pool labelling: _select → predict → reveal label → update_.
- **Metric:** Cumulative errors vs. number of labeled samples.
- **Tracks:** Cross of initialization (**Scratch** vs **Pretrained**) and update schedule
  (**Online** vs **Batch**).
- **Scope:** Vision (MNIST, Fashion‑MNIST, CIFAR‑10, SVHN) and Text (AG News).
- **Goal:** Reproducible comparison of models and strategies in the early phase.

---

## 2. Protocol
At each step _t_:
1) Select an unlabeled instance from the pool (by a strategy).
2) Predict its label.
3) Obtain the true label from the oracle.
4) Update model parameters (per‑sample _Online_ or every _K_ samples in _Batch_).

**No abstention:** every instance must be predicted.
**Primary output:** the error trajectory and its final value.

---

## 3. Tracks
Four tracks isolate the effects of prior knowledge and update cadence:

- **G&L–SO (Scratch–Online):** random initialization; update **after every sample**.
- **G&L–SB_K (Scratch–Batch):** random initialization; **batch** update every **K>1** samples.
- **G&L–PO (Pretrained–Online):** start from pretrained weights; update **after every sample**.
- **G&L–PB_K (Pretrained–Batch):** pretrained; **batch** update every **K>1** samples.

**Naming:** `...-SB_50` and `...-PB_50` denote `K=50`. If no suffix is present, `K=1` (online).

The driver enforces sensible pairings (scratch models on scratch tracks; pretrained models on
pretrained tracks).

---

## 4. Models & Datasets
Supported combinations:

### Vision dataset (MNIST)
- **Scratch:** `knn`, `perceptron`, `cnn`
- **Pretrained:** `resnet50`, `vit-b-16`
  (ViT maps to `google/vit-base-patch16-224-in21k`; ResNet uses ImageNet weights.)

### Text dataset (AG News)
- **Scratch:** `text-knn` (k‑NN on TF‑IDF), `text-perceptron` (linear on TF‑IDF)
- **Pretrained:** `bert-base` (maps to `bert-base-uncased`)

Vision inputs are normalized for pretrained models (resize to 224×224 if needed, grayscale
expanded to 3 channels). Text models use TF‑IDF (scratch) or the BERT tokenizer (pretrained).

---

## 5. Acquisition strategies
- **random** — uniform sampling (baseline)
- **confidence** — highest max class probability (easy‑first)
- **least_confidence** — lowest max class probability (uncertainty‑first)
- **margin** — smallest (top‑1 − top‑2) probability gap
- **entropy** — highest predictive entropy

These strategies shape which examples are seen early, affecting the error trajectory.

---

## 6. Installation
Prefer **requirements.txt** for a consistent install (handles pinned packages and the
appropriate PyTorch wheel). As an alternative, **Pipenv** files (Pipfile and Pipfile.lock) are
provided.

### 6.1. Using requirements.txt (preferred)
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
> **Note on PyTorch wheels:** If you install manually, ensure you use the correct
> CUDA/CPU/MPS build and, if needed, the PyTorch extra index URL. Using
> `requirements.txt` avoids most of that friction.

### 6.2. Using Pipenv (alternative)
Use the lockfile for a reproducible environment:
```bash
pip install --upgrade pip pipenv
pipenv sync               # create from Pipfile.lock
pipenv shell              # activate
```
If you prefer to resolve from Pipfile:
```bash
pipenv install
pipenv shell
```

---

## 7. Running experiments
Entry point: `src/run_all.py`.

### 7.1. Full default grid
```bash
python src/run_all.py --all
```
Runs default datasets, models, strategies, and tracks with seeds `[0,1,2]` and saves to `./results`.

### 7.2. Selective runs
```bash
python src/run_all.py   --datasets mnist,ag_news   --models perceptron,cnn,bert-base   --strategies random,entropy   --tracks G&L-SO,G&L-SB_50,G&L-PB_50   --seeds 0 1 2   --output_dir results/exp1
```

### 7.3. Useful flags (summary)
- `--datasets / --models / --strategies / --tracks` — comma‑separated lists
- `--seeds` — one or more ints (e.g., `--seeds 0 1 2`)
- `--devices` — default: `cuda` if available; else `mps` (Apple); else `cpu`
- `--workers` — parallel processes (default: up to 8)
- `--output_dir` — results folder (default: `results`)
- `--subset N` — cap pool size globally (e.g., 300)
- `--subset_map "mnist:1000,ag_news:2000"` — per‑dataset caps
- `--reset-weights` — for batch tracks, reinit weights before each batch update
- `--shard k n` — run only shard `k` of `n` across the grid

**Idempotent skipping:** the plot image is treated as the final artifact. If it exists and is
valid, the run is skipped. With `RESULTS_S3_PREFIX` set (see below), the same applies to S3.

---

## 8. Outputs
For each configuration `{dataset}_{model}_{strategy}_{track}_seed{S}[_s{subset}]`:
- `*_results.json` — error trajectory, final metrics, params, labeled indices, per‑step
  error flags, duration, and RNG states.
- `*_plot.png` — cumulative errors vs. labeled samples (with final error annotation).
- `*_labels.pt` — tensor of ground‑truth labels for the pool.
- `*_features.pt` — learned features if the model implements `extract_features`;
  otherwise flattened raw features for tensor vision data; text models log when not applicable.

These artifacts are sufficient for post‑hoc analysis and plotting without rerunning training.

---

## 9. Reproducibility and settings
- **Seeding:** Python, NumPy, and PyTorch are seeded per run; seeds appear in filenames
  and JSON.
- **Determinism:** deterministic algorithms are requested where available; cuDNN
  benchmarking is disabled to reduce nondeterminism.
- **Device selection:** default is `cuda` → `mps` → `cpu`, or override via `--devices`.
- **File identity:** filenames encode dataset/model/strategy/track/seed/subset.

---

## 10. Optional S3 cache
Set an S3 prefix to enable remote caching of finished artifacts:
```bash
export RESULTS_S3_PREFIX=s3://my-bucket/gnl-runs
```
If enabled, completed results are uploaded, and future runs will download/verify and skip
matching experiments.

---

## 11. White paper
This repository implements the methodology described in the paper:

> **“Guess‑and‑Learn (G&L): Measuring the Cumulative Error Cost of Cold‑Start Adaptation.”**

A public link will be added once available. The paper motivates cumulative‑error evaluation,
defines the four tracks, and analyzes empirical results relevant to early‑phase learning.
