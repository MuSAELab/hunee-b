import os
import glob
import random
import logging
import argparse
import numpy as np

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

from beessl.hub import options
from beessl.downstream.runner import Runner


def get_downstream_args():
    parser = argparse.ArgumentParser()

    # train or test for this experiment
    parser.add_argument('-m', '--mode', choices=['train', 'evaluate'], required=True)
    parser.add_argument('-o', '--override', help='Used to override args and config, this is at the highest priority')

    # @TODO: Add re-initialization scheme to continue training from a checkpoint
    # Downstream settings
    parser.add_argument('-d', '--downstream', help='\
        Typically downstream dataset need manual preparation.\
        Please check downstream/README.md for details'
    )

    # Upstream settings
    upstreams = options()
    parser.add_argument('-u', '--upstream',  help=""
        'Upstreams need a local ckpt (-k) or config file (-g). '
        'Please check upstream/README.md for details. '
        f"Available options in BEESSL: {upstreams}."
    )
    parser.add_argument('-k', '--upstream_ckpt', help='Path to the upstream ckpt')
    parser.add_argument('-g', '--upstream_model_config', help='The config file for constructing the pretrained model')
    parser.add_argument('-s', '--upstream_feature_selection', default='hidden_states', help='Specify the layer to be extracted as the representation')
    parser.add_argument('-l', '--upstream_layer_selection', type=int, help='Select a specific layer for the features selected by -s')
    parser.add_argument('--upstream_feature_normalize', action='store_true', help='Specify whether to normalize hidden features before weighted sum')

    # experiment directory, choose one to specify
    # expname uses the default root directory: result/downstream
    parser.add_argument('-n', '--expname', help='Save experiment at result/downstream/expname')
    parser.add_argument('-p', '--expdir', help='Save experiment at expdir')
    parser.add_argument('-a', '--auto_resume', action='store_true', help='Auto-resume if the expdir contains checkpoints')

    # options
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')

    args = parser.parse_args()
    if args.expdir is None:
        args.expdir = f'result/downstream/{args.expname}'

    print('[Runner] - Start a new experiment')
    os.makedirs(args.expdir, exist_ok=True)

    args.config = f'./downstream/{args.downstream}/config.yaml'
    with open(args.config, 'r') as f:
        config = load_hyperpyyaml(f, args.override)

    return args, config


def seed_everything(seed):
    # Fix seed and make backends deterministic
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    logging.basicConfig(level=logging.INFO)

    # get config and arguments
    args, config = get_downstream_args()
    seed_everything(args.seed)

    runner = Runner(args, config)
    eval(f'runner.{args.mode}')()


if __name__ == '__main__':
    main()