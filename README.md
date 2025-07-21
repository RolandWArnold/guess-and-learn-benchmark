Guess-and-Learn: A Diagnostic Benchmark for Zero-Shot Error Efficiency
This repository contains the official reference implementation for the paper "Guess-and-Learn: A Diagnostic Benchmark for Zero-Shot Error Efficiency".

Overview
Guess-and-Learn (G&L) is a protocol for measuring a model's error-efficiency. It quantifies how many mistakes a model makes while sequentially labeling an entire dataset from a cold start (zero in-domain labels). This benchmark is designed to complement existing metrics like accuracy and label-efficiency by focusing on the "cost of learning" in terms of errors.

This implementation provides the tools to run the G&L protocol across the four defined tracks (SO, SB, PO, PB) with various models, datasets, and acquisition strategies.

Installation
Clone the repository:bash
git clone https://github.com/your-username/guess-and-learn.git
cd guess-and-learn


Install the required dependencies:

Bash

pip install -r requirements.txt
Running Experiments
The main entry point for running experiments is the scripts/run_experiment.py script. It uses command-line arguments to configure the G&L run.

Example Commands
1. Run Perceptron on MNIST (G&L-SO track) with Random Strategy:

Bash

python scripts/run_experiment.py \
    --dataset mnist \
    --model perceptron \
    --strategy random \
    --track G\&L-SO \
    --seed 1
2. Run a 3-layer CNN on CIFAR-10 (G&L-SB track) with Entropy Strategy:
This uses a batch update every K=50 samples.

Bash

python scripts/run_experiment.py \
    --dataset cifar10 \
    --model cnn \
    --strategy entropy \
    --track G\&L-SB \
    --k_batch 50 \
    --lr 0.01 \
    --epochs_per_update 5 \
    --seed 1
3. Run ViT-B/16 Fine-tuning on CIFAR-10 (G&L-PB track) with Margin Strategy:
This uses a batch update every K=200 samples. Note that ViT requires resizing, which should be handled in a more advanced datasets.py for real use (e.g., via torchvision.transforms.Resize).

Bash

python scripts/run_experiment.py \
    --dataset cifar10 \
    --model vit-b-16 \
    --strategy margin \
    --track G\&L-PB \
    --k_batch 200 \
    --lr 2e-5 \
    --epochs_per_update 3 \
    --seed 1
Running Multiple Seeds
To generate the mean and standard deviation plots shown in the paper, you should run the same command multiple times with different --seed values.

Bash

for i in {1..10}
do
   python scripts/run_experiment.py \
       --dataset cifar10 \
       --model cnn \
       --strategy entropy \
       --track G\&L-SB \
       --k_batch 50 \
       --seed $i
done
Visualizing Results
After running experiments, the results (JSON files and plots for each run) will be saved in the --output_dir (default: ./results).

You can use the notebooks/visualize_results.ipynb Jupyter notebook to aggregate results from multiple seeds and generate the comparative plots presented in the paper.

Launch Jupyter Lab or Jupyter Notebook:

Bash

jupyter lab
Open notebooks/visualize_results.ipynb and run the cells. The notebook is configured to automatically find result files in the results directory, group them by experiment configuration, and plot the mean error curves with standard deviation bands.

Extending the Framework
The framework is designed to be modular:

To add a new dataset: Add a new case to get_dataset in guess_and_learn/datasets.py.

To add a new model: Create a new class inheriting from GnlModel in guess_and_learn/models.py and add it to the get_model factory function.

To add a new acquisition strategy: Create a new class inheriting from AcquisitionStrategy in guess_and_learn/strategies.py and add it to the get_strategy factory function.