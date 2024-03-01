from hyperpyyaml import load_hyperpyyaml
from beessl.upstream.ssast.expert import UpstreamExpert

def ssast(*args, **kwargs):
    if kwargs["model_config"]:
        model_config = kwargs["model_config"]
    else:
        raise ValueError("model_config (-g) is required")

    with open(model_config, 'r') as f:
        config = load_hyperpyyaml(f)

    return UpstreamExpert(config, ckpt=kwargs["ckpt"])