#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${OUTDIR:-./runs}"
DEVICES="${DEVICES:-cpu}"   # set DEVICES=cuda to use GPU if available

run() {
  echo "+ $*"
  python src/run_all.py "$@" --output_dir "$OUTDIR" --devices "$DEVICES"
}

echo "== AG News — BERT-BASE (PO & PB_50) =="

# Subset 300 for all strategies on PO and PB_50, seeds 0..4
run --datasets ag_news --models bert-base \
    --strategies random confidence least_confidence margin entropy \
    --tracks 'G&L-PO' 'G&L-PB_50' \
    --seeds 0 1 2 3 4 \
    --subset 300

# Full runs (no subset) for random & entropy on PB_50
run --datasets ag_news --models bert-base \
    --strategies random entropy \
    --tracks 'G&L-PB_50' \
    --seeds 0 1 2 3 4

# “reset” variant for PB_50 + entropy + subset 300
run --datasets ag_news --models bert-base \
    --strategies entropy \
    --tracks 'G&L-PB_50' \
    --seeds 0 1 2 3 4 \
    --subset 300 \
    --reset

echo "== AG News — text-knn & text-perceptron (SO & SB_50) =="

# Subset 300 for all strategies on SO & SB_50, seeds 0..4
run --datasets ag_news --models text-knn text-perceptron \
    --strategies random confidence least_confidence margin entropy \
    --tracks 'G&L-SO' 'G&L-SB_50' \
    --seeds 0 1 2 3 4 \
    --subset 300

# Full runs (no subset) for text-perceptron on SO, all strategies
run --datasets ag_news --models text-perceptron \
    --strategies random confidence least_confidence margin entropy \
    --tracks 'G&L-SO' \
    --seeds 0 1 2 3 4

echo "== Fashion-MNIST — knn (SO & SB_50) =="
run --datasets fashion-mnist --models knn \
    --strategies random \
    --tracks 'G&L-SO' 'G&L-SB_50' \
    --seeds 0 1

echo "== MNIST — cnn (SO & SB_K where K∈{10,50,200}) =="

# Subset 300 for all strategies, all K, seeds 0..4
run --datasets mnist --models cnn \
    --strategies random confidence least_confidence margin entropy \
    --tracks 'G&L-SO' 'G&L-SB_10' 'G&L-SB_50' 'G&L-SB_200' \
    --seeds 0 1 2 3 4 \
    --subset 300

# Full runs (no subset) for SB_50 with random & entropy
run --datasets mnist --models cnn \
    --strategies random entropy \
    --tracks 'G&L-SB_50' \
    --seeds 0 1 2 3 4

echo "== MNIST — knn & perceptron (SO & SB_50) =="

# Subset 300 for all strategies, seeds 0..4
run --datasets mnist --models knn perceptron \
    --strategies random confidence least_confidence margin entropy \
    --tracks 'G&L-SO' 'G&L-SB_50' \
    --seeds 0 1 2 3 4 \
    --subset 300

# Full runs (no subset) for random on SO & SB_50
run --datasets mnist --models knn perceptron \
    --strategies random \
    --tracks 'G&L-SO' 'G&L-SB_50' \
    --seeds 0 1 2 3 4

echo "== MNIST — resnet50 & vit-b-16 (PO & PB_50) =="

# Subset 300 (all strategies) on PO & PB_50, seeds 0..4
run --datasets mnist --models resnet50 vit-b-16 \
    --strategies random confidence least_confidence margin entropy \
    --tracks 'G&L-PO' 'G&L-PB_50' \
    --seeds 0 1 2 3 4 \
    --subset 300

# Full runs (no subset) for random & entropy on PB_50
run --datasets mnist --models resnet50 vit-b-16 \
    --strategies random entropy \
    --tracks 'G&L-PB_50' \
    --seeds 0 1 2 3 4

echo "All jobs submitted."
