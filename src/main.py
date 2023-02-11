"""This is the main entry point for the application."""
import argparse

import torch

from app import run_app

# get cli parameters
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--disable_training",
    type=int,
    default=1,
    required=False,
    help="Boolean, whether to diable training app or not",
)

if __name__ == "__main__":
    args = parser.parse_args()
    disable_training = True if args.disable_training else False
    print("CUDA Found: ", torch.cuda.is_available())
    run_app(disable_training=disable_training)
