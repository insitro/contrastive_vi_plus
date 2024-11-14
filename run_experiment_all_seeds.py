import argparse
import os

from constants import (
    DEFAULT_RANDOM_SEEDS,
    METHODS,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--method", choices=METHODS)
parser.add_argument(
    "--inference_strategy", type=str, choices=["marginalize", "gumbel_sigmoid"]
)
parser.add_argument("--n_classifier_layers", type=int, default=3)
parser.add_argument("--mmd_penalty", type=float, default=0)
parser.add_argument("--learn_basal_mean", action="store_true")
parser.add_argument("--early_stopping", action="store_true")

args = parser.parse_args()

for seed in DEFAULT_RANDOM_SEEDS:
    command_string = f"python run_experiment.py --dataset={args.dataset} --method={args.method} --seed={seed} --n_classifier_layers={args.n_classifier_layers} --mmd_penalty={args.mmd_penalty} --inference_strategy={args.inference_strategy}"

    if args.learn_basal_mean:
        command_string += " --learn_basal_mean"
    if args.early_stopping:
        command_string += " --early_stopping"

    os.system(command_string)
