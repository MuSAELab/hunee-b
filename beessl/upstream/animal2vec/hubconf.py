from beessl.upstream.aves.expert import UpstreamExpert


def animal2vec(*args, **kwargs):
    return UpstreamExpert(ckpt=kwargs["ckpt"])
