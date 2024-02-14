import importlib
from beessl import hub

class Runner:
    def __init__(self, args, config):
        self.args = args
        self.config = config

        # Load downstream components
        self.upstream_model = self._get_upstream()
        self.pretrain = self._get_pretrain()

    def _get_upstream(self):
        Upstream = getattr(hub, self.args.upstream)
        ckpt_path = self.args.upstream_ckpt
        print(f"[Runner] - Loading upstream {self.args.upstream.upper()}")
        if ckpt_path:
            print(f"[Runner] - Loading ckpt from {ckpt_path}")

        return Upstream(
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
        ).to(self.args.device)

    def _get_pretrain(self):
        brain = importlib.import_module(
            f"beessl.pretrain.{self.args.upstream}.brain"
        )

        return getattr(brain, "PretrainBrain")(
            upstream=self.upstream_model,
            args=self.args,
            config=self.config
        )

    def train(self):
        self.pretrain.pretrain_upstream()
