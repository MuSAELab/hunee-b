from beessl.upstream.byola.expert import UpstreamExpert
from beessl.upstream.byola.common import load_yaml_config

def byola(*args, **kwargs):
    if kwargs["model_config"]:
        model_config = kwargs["model_config"]
    else:
        raise ValueError("model_config (-g) is required")

    config = load_yaml_config(model_config)
    return UpstreamExpert(config, ckpt=kwargs["ckpt"])
