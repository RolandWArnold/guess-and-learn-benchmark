import argparse
import random
import numpy as np
import torch
import itertools
import multiprocessing
from guess_and_learn.datasets import get_data_for_protocol
from guess_and_learn.models import get_model
from guess_and_learn.strategies import get_strategy
from guess_and_learn.protocol import GnlProtocol, save_results


def run_single_experiment(args):
    seed, dataset, model_name, strategy_name, track, K, device = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    X, Y = get_data_for_protocol(dataset)
    model = get_model(model_name, dataset, device)
    strategy = get_strategy(strategy_name)

    track_config = {'track': track, 'K': K}
    if 'pretrained' in model_name:
        track_config.update({'lr': 2e-5, 'epochs_per_update': 3, 'train_batch_size': 16})
    elif model_name == 'cnn':
        track_config.update({'lr': 0.01, 'epochs_per_update': 5, 'train_batch_size': 32})

    protocol = GnlProtocol(model, strategy, X, Y, track_config)
    results = protocol.run()
    params = {
        'seed': seed,
        'dataset': dataset,
        'model': model_name,
        'strategy': strategy_name,
        'track': track
    }
    save_results(results, params, output_dir='results')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Run all combinations')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--devices', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.devices)

    datasets = ['mnist', 'fashion-mnist', 'cifar10', 'svhn', 'ag_news']
    models = ['knn', 'perceptron', 'cnn', 'resnet50', 'vit-b-16', 'bert-base']
    strategies = ['random', 'confidence', 'least_confidence', 'margin', 'entropy']
    tracks_k = [('G&L-SO', 1), ('G&L-SB_K', 50), ('G&L-PO', 1), ('G&L-PB_K', 200)]

    if args.all:
        combinations = list(itertools.product(
            args.seeds, datasets, models, strategies, tracks_k
        ))

        expanded_combinations = []
        for seed, ds, model, strat, (track, K) in combinations:
            if ds == 'ag_news' and model not in ['bert-base']:
                continue  # skip incompatible
            if ds != 'ag_news' and model == 'bert-base':
                continue
            expanded_combinations.append((seed, ds, model, strat, track, K, device))

        with multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count())) as pool:
            pool.map(run_single_experiment, expanded_combinations)

    else:
        print("Use --all to run full experiment suite")


if __name__ == '__main__':
    main()
