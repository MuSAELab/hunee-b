import os
import random
import logging
import argparse
import numpy as np

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

from beessl import hub
from beessl.hub import options
from beessl.pretrain.runner import Runner


def get_pretrain_args():
    parser = argparse.ArgumentParser()

    # train or test for this experiment
    parser.add_argument('-o', '--override', help='Used to override args and config, this is at the highest priority')

    # Upstream settings
    upstreams = options()
    parser.add_argument('-u', '--upstream',  help=""
        'Upstreams need a local ckpt (-k) or config file (-g). '
        'Please check upstream/README.md for details. '
        f"Available options in BEESSL: {upstreams}."
    )
    parser.add_argument('-k', '--upstream_ckpt', help='Path to the upstream ckpt')
    parser.add_argument('-g', '--upstream_model_config', help='The config file for constructing the pretrained model')

    # experiment directory, choose one to specify
    parser.add_argument('-p', '--expdir', help='Save experiment at expdir')

    # options
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')

    args = parser.parse_args()
    if args.expdir is None:
        args.expdir = f'result/pretrain/{args.upstream}'

    print('[Runner] - Start a new experiment')
    os.makedirs(args.expdir, exist_ok=True)

    args.config = f'./pretrain/{args.upstream}/config_runner.yaml'
    args.override = parse_override(args)
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

def parse_override(args):
    num_hidden_layers, in_features = get_representation_info(args)
    override = f"expdir={args.expdir},,"\
               f"num_hidden_layers={num_hidden_layers},,"\
               f"in_features={in_features}"

    if args.override is not None:
        override = f"{override},,{args.override}"

    # parse override
    override = override.split(',,')
    override = [kv.replace('=', ': ') for kv in override]
    override = "\n".join(override)

    return override

def get_representation_info(args):
    Upstream = getattr(hub, args.upstream)
    model = Upstream(
        ckpt = args.upstream_ckpt,
        model_config = args.upstream_model_config
    ).to(args.device)

    fake_input = torch.randn(1, 16000).to(args.device)
    with torch.no_grad():
        fake_output = model(fake_input)["hidden_states"]

    num_hidden_layers = len(fake_output)
    in_features = fake_output[0].shape[1]
    return num_hidden_layers, in_features

def main():
    logging.basicConfig(level=logging.INFO)

    # get config and arguments
    args, config = get_pretrain_args()
    seed_everything(args.seed)

    runner = Runner(args, config)
    runner.train()


if __name__ == '__main__':
    main()