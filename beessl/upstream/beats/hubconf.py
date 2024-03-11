from beessl.upstream.beats.expert import UpstreamExpert

def beats(*args, **kwargs):
    return UpstreamExpert(ckpt=kwargs["ckpt"])