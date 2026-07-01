import argparse

from uchiha.apis import set_random_seed
from uchiha.apis.inference_dehaze import single_inference
from uchiha.utils import load_config


def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--seed', type=int, default=49)
    args_parser.add_argument('--config', type=str, default='configs/hsi_dehaze/HD/inference/D3.yaml')
    return args_parser.parse_args()


def inference():
    args = parse_args()
    cfg = load_config(args.config)
    set_random_seed(args.seed)
    single_inference(cfg)


if __name__ == '__main__':
    inference()
