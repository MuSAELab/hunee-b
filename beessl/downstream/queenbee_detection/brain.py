import torch
import speechbrain as sb
import torch.nn.functional as F
import speechbrain.nnet.schedulers as schedulers
from beessl.downstream.queenbee_detection.dataset import dataio_prep
from beessl.downstream.queenbee_detection.dataset import prepare_nuhive
from sklearn.metrics import roc_auc_score

from collections import defaultdict

def roc_auc_segments(error_metric):
    ids = error_metric.ids
    y_true = error_metric.labels.cpu()
    y_pred = error_metric.scores.cpu()

    y_pred = y_pred.sigmoid()
    ids = [i.split("_chunk_")[0] for i in ids]

    # Get the index of repeated elements
    pos_map = defaultdict(list)
    for pos, ele in enumerate(ids):
        pos_map[ele].append(pos)

    ids_keys = pos_map.keys()
    y_pred_agg = [y_pred[pos_map[i]].mean() for i in ids_keys]
    y_true_agg = [y_true[pos_map[i]].mean() for i in ids_keys]

    return roc_auc_score(y_true_agg, y_pred_agg)


class DownstreamBrain(sb.Brain):
    def __init__(self, upstream, args, config):
        self.args = args
        self.config = config
        self.upstream = upstream

        super().__init__(
            modules=config["modules"],
            opt_class=config["opt_class"],
            hparams=config,
            run_opts={"device": self.args.device},
            checkpointer=config["checkpointer"],
        )

        self.prepare_datasets()

    def compute_forward(self, batch, stage):
        signal, lens = batch.bee_sig
        signal = signal.to(self.device) # B x T
        signal = signal / signal.max(dim=-1, keepdim=True)[0] # Normalize the signal

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN and self.hparams.augmentation is not None:
            lens = lens.to(self.device)
            signal = self.hparams.augmentation(signal, lens)

        # Extract features from upstream and weight them
        # Nomenclature: (L = Layers, F = Features, T = Time)
        feats = self.upstream(signal)["hidden_states"] # L x [B x F x T] (List of tensors)
        feats = self.modules.featurizer(feats) # B x F x T

        # Attention pooling and perform the classification
        feats = self.modules.pooling(feats).squeeze(-1) # B x F x T => B x F
        return self.modules.projector(feats)

    def compute_objectives(self, predictions, batch, stage):
        # Get targets
        targets, lens = batch.target
        targets = targets.to(self.device) # B, 1

        # Compute the loss function
        loss = self.hparams.loss(predictions, targets)
        if (stage != sb.Stage.TRAIN):
            self.error_metric.append(
                ids=batch.id,
                scores=predictions,
                labels=targets
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=self.hparams.loss
        )

        # Add a metric for evaluation sets
        if stage != sb.Stage.TRAIN:
            self.error_metric = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        if stage != sb.Stage.TRAIN:
            # Summarize the statistics from the stage for record-keeping.
            metrics = self.error_metric.summarize()
            roc_auc = roc_auc_segments(self.error_metric)

            stats = {
                "loss": stage_loss,
                "f-score": metrics["F-score"],
                "roc_auc": roc_auc,
            }

        # At the end of validation, we can write stats, checkpoints and update LR.
        if (stage != sb.Stage.TRAIN):
            if (stage == sb.Stage.VALID):
                current_lr, next_lr = self.hparams.lr_scheduler(stage_loss)
                schedulers.update_learning_rate(self.optimizer, next_lr)

                # The train_logger writes a summary to stdout and to the logfile.
                self.hparams.train_logger.log_stats(
                    stats_meta={"Epoch": epoch, "LR": current_lr},
                    train_stats={"loss": self.train_loss},
                    valid_stats=stats,
                )

                # Save the current checkpoint and delete previous checkpoints,
                # unless they have the current best task1_metric
                self.checkpointer.save_and_keep_only(meta=stats, max_keys=["f-score"])

            elif stage == sb.Stage.TEST:
                self.hparams.train_logger.log_stats(
                    stats_meta={"Epoch Loaded": self.hparams.epoch_counter.current},
                    test_stats=stats,
                )

    def train_downstream(self):
        print("[DownstreamBrain] - Starting the training process")
        self.fit(
            epoch_counter=self.hparams.epoch_counter,
            train_set=self.datasets["train"],
            valid_set=self.datasets["valid"],
            train_loader_kwargs=self.hparams.dataloader_options,
            valid_loader_kwargs=self.hparams.dataloader_options
        )

    def evaluate_downstream(self):
        print("[DownstreamBrain] - Starting the evaluate process")
        self.evaluate(
            test_set=self.datasets["test"],
            max_key="f-score",
            progressbar=True,
            test_loader_kwargs=self.hparams.dataloader_options,
        )

    def prepare_datasets(self):
        prepare_nuhive(
            data_folder=self.config["data_root"],
            save_folder=self.config["annotation_folder"],
            skip_prep=False,
            extension=".wav",
            chunkwise=self.config["chunkwise"],
            chunk_length=self.config["chunk_length"],
        )
        self.datasets = dataio_prep(self.config)
