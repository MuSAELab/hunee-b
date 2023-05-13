from hyperpyyaml import load_hyperpyyaml
from beessl.upstream.baseline.expert import UpstreamExpert

def spectrogram(*args, **kwargs):
    model_config = "upstream/baseline/hparams/spec.yaml"
    if kwargs["model_config"]:
        model_config = kwargs["model_config"]

    with open(model_config, 'r') as f:
        config = load_hyperpyyaml(f)
    return UpstreamExpert(config)


def mfcc(*args, **kwargs):
    model_config = "upstream/baseline/hparams/mfcc.yaml"
    if kwargs["model_config"]:
        model_config = kwargs["model_config"]

    with open(model_config, 'r') as f:
        config = load_hyperpyyaml(f)
    return UpstreamExpert(config)


def fbank(*args, **kwargs):
    model_config = "upstream/baseline/hparams/fbank.yaml"
    if kwargs["model_config"]:
        model_config = kwargs["model_config"]

    with open(model_config, 'r') as f:
        config = load_hyperpyyaml(f)
    return UpstreamExpert(config)
