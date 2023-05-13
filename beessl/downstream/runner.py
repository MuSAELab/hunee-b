import torch
from beessl import hub
from beessl.downstream.featurizer import Featurizer

class Runner:
    def __init__(self, args, config):
        self.args = args
        self.config = config

        # Load downstream components
        self.upstream_model = self._get_upstream()
        self.featurizer = self._get_featurizer()
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

    def _get_featurizer(self):
        return Featurizer()

    def _get_downstream(self):
        pass

    def train(self):
        print("[Runner] - Starting the training process")

    def evaluate(self):
        print("[Runner] - Starting the evaluate process")