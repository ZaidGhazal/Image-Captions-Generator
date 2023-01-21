from app import run_app
import argparse

# get cli parameters
parser = argparse.ArgumentParser(description='')
parser.add_argument('--disable_training', type=bool, default=True, required=False,
                    help='Boolean, whether to diable training app or not')
if __name__ == "__main__":
    args = parser.parse_args()
    disable_training = args.disable_training
    print("disable training: ", disable_training)
    run_app(disable_training=disable_training)