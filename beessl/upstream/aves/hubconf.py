from hyperpyyaml import load_hyperpyyaml
from beessl.upstream.aves.expert import UpstreamExpert


def aves(*args, **kwargs):
    model_config = "upstream/aves/hparams/aves.yaml"
    if kwargs["model_config"]:
        model_config = kwargs["model_config"]

    with open(model_config, "r") as f:
        config = load_hyperpyyaml(f)
    return UpstreamExpert(config, ckpt=kwargs["ckpt"])


def birdaves(*args, **kwargs):
    model_config = "upstream/aves/hparams/birdaves.yaml"
    if kwargs["model_config"]:
        model_config = kwargs["model_config"]

    with open(model_config, "r") as f:
        config = load_hyperpyyaml(f)
    return UpstreamExpert(config, ckpt=kwargs["ckpt"])
