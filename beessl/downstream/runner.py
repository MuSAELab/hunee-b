import torch
import importlib
from beessl import hub
from beessl.downstream.featurizer import Featurizer

class Runner:
    def __init__(self, args, config):
        self.args = args
        self.config = config

        # Load downstream components
        self.upstream_model = self._get_upstream()
        n_params = sum(p.numel() for p in self.upstream_model.parameters())
        print(f"[Runner] - Number of upstream parameters {n_params}")
        self.downstream = self._get_downstream()

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

    def _get_downstream(self):
        print(f"[Runner] - Selected the {self.args.downstream} downstream")
        brain = importlib.import_module(
            f"beessl.downstream.{self.args.downstream}.brain"
        )

        return getattr(brain, "DownstreamBrain")(
            upstream=self.upstream_model,
            args=self.args,
            config=self.config
        )

    def train(self):
        self.downstream.train_downstream()

    def evaluate(self):
        self.downstream.evaluate_downstream()