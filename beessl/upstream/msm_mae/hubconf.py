from beessl.upstream.msm_mae.expert import UpstreamExpert

def msm_mae(*args, **kwargs):
    return UpstreamExpert(ckpt=kwargs["ckpt"])