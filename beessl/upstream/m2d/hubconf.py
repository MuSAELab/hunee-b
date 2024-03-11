from beessl.upstream.m2d.expert import UpstreamExpert

def m2d(*args, **kwargs):
    return UpstreamExpert(ckpt=kwargs["ckpt"])