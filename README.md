# Guess-and-Learn: A Diagnostic Benchmark for Zero-Shot Error Efficiency

This repository contains the official reference implementation for the paper "Guess-and-Learn: A Diagnostic Benchmark for Zero-Shot Error Efficiency".

## Overview

**Guess-and-Learn (G&L)** is a diagnostic protocol for measuring a learning algorithm's cold-start error efficiency. It quantifies the cumulative number of prediction mistakes a model makes while sequentially labeling an entire unlabeled dataset, starting from zero in-domain labeled examples.

At each step, the protocol is as follows:
1. **Select**: The model uses an acquisition strategy to select an unlabeled point from a pool.
2. **Guess**: The model predicts the label for the selected point.
3. **Oracle**: The ground-truth label is revealed.
4. **Update Error**: The cumulative error count is incremented if the guess was incorrect.
5. **Update State**: The model updates its internal state using the newly acquired label, according to the rules of the specific track.

The resulting cumulative error trajectory serves as a transparent measure of a model's inductive bias, adaptation speed, and sample-selection strategy quality. This benchmark complements classic metrics like accuracy and label-efficiency by focusing on the "cost of learning" in terms of errors, which is critical in high-stakes or interactive settings.

The benchmark is organized into four tracks to disentangle the influence of inductive bias, representation learning, and feature transfer:
* **G&L-SO (Scratch-Online)**: Train from scratch with online (single-example) updates.
* **G&L-SB (Scratch-Batch)**: Train from scratch with batch updates every *K* samples.
* **G&L-PO (Pretrained-Online)**: Fine-tune a pretrained model with online updates (typically on the classifier head only).
* **G&L-PB (Pretrained-Batch)**: Fine-tune a pretrained model with batch updates every *K* samples.

## 1. Installation and Setup

### Clone the Repository
```bash
git clone https://github.com/RolandWArnold/guess-and-learn.git
cd guess-and-learn
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Prepare Caches (Recommended)

To run experiments fully offline and ensure reproducibility, you can pre-download all required datasets (e.g., MNIST, CIFAR-10, AG News) and pretrained model weights (e.g., ViT, BERT, ResNet-50).

```bash
python scripts/prepare_cache.py
```

This script will populate a local cache, which is essential for environments without network access.

## 2. Running Experiments

The main entry point for running experiments is the `run_all.py` script. It allows you to define a grid of experiments and leverages multiprocessing to execute them in parallel.

### Running a Single Experiment

You can run a single experiment by specifying one option for each dimension (dataset, model, strategy, track, and seed).

**Example: Run Perceptron on MNIST (G&L-SO track) with the Random strategy.**

```bash
./run_all.py \
    --datasets mnist \
    --models perceptron \
    --strategies random \
    --tracks "G&L-SO" \
    --seeds 1 \
    --workers 1
```

### Running an Experiment Grid

The script's power comes from its ability to launch a grid of experiments. You can provide comma-separated lists of values.

**Example: Run a 3-layer CNN and ViT-B/16 on CIFAR-10, using two different strategies and three seeds.**
The batch size `K` is automatically parsed from the track name (e.g., `G&L-PB_50` sets K=50).

```bash
./run_all.py \
    --datasets cifar10 \
    --models cnn,vit-b-16 \
    --strategies entropy,margin \
    --tracks "G&L-SB_50,G&L-PB_50" \
    --subset 300 \
    --seeds 1 2 3 \
    --workers 8
```

### Key Arguments

* `--datasets`: Comma-separated list of datasets (e.g., `mnist,cifar10,ag_news`).
* `--models`: Comma-separated list of models (e.g., `cnn,vit-b-16,bert-base`). The script automatically filters for valid model/dataset pairs (e.g., BERT runs only on AG News).
* `--strategies`: Comma-separated list of acquisition strategies (e.g., `random,entropy,least_confidence`).
* `--tracks`: Comma-separated list of G&L tracks (e.g., `"G&L-SO,G&L-SB_50"`).
* `--seeds`: Space-separated list of random seeds for reproducibility (e.g., `1 2 3 4 5`).
* `--subset`: Restrict the dataset to a smaller random subset of a given size (e.g., `300`). This is useful for rapid testing.
* `--reset-weights`: A special flag for batch tracks that re-initializes model weights from scratch before each batch update. This helps isolate pure sample-efficiency.
* `--workers`: Number of parallel processes to use.
* `--output_dir`: Directory to save results (defaults to `./results`).

## 3. Visualizing and Analyzing Results

### Per-Run Plots

For each experiment, a plot of the cumulative error curve is automatically saved in the output directory (e.g., `./results/mnist_cnn_entropy_G&L-SB_50_seed1_plot.png`).

### Aggregate Plots

To aggregate results from multiple seeds and generate the comparative plots shown in the paper (with mean and standard deviation bands), use the provided Jupyter notebook.

1. Launch Jupyter Lab or Jupyter Notebook:
   ```bash
   jupyter lab
   ```
2. Open `notebooks/visualize_results.ipynb` and run the cells. The notebook automatically finds result files, groups them by experiment configuration, and generates the final plots.

### Check Benchmark Completeness

To verify if you have run all experiments for a complete benchmark (e.g., for the `s300` subset across 5 seeds), you can use the `check_completeness.py` script. It will report any missing runs and generate the precise `run_all.py` commands needed to complete them.

```bash
python scripts/check_completeness.py ./results --subset 300
```

## 4. Extending the Framework

The framework is designed to be modular and easy to extend.

* **Add a new dataset**: Add a new case to `get_dataset` and `get_data_for_protocol` in `guess_and_learn/datasets.py`.
* **Add a new model**: Create a new class inheriting from `GnlModel` in `guess_and_learn/models.py` and add it to the `get_model` factory function.
* **Add a new acquisition strategy**: Create a new class inheriting from `AcquisitionStrategy` in `guess_and_learn/strategies.py` and add it to the `get_strategy` factory function.