import copy
import torch
import torch.nn.functional as F
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers

from beessl.pretrain.beeyol.byol import EMA
from beessl.pretrain.beeyol.dataset import dataio_prep
from beessl.pretrain.beeyol.dataset import prepare_nectar



class PretrainBrain(sb.Brain):
    def __init__(self, upstream, args, config):
        self.args = args
        self.config = config

        config["modules"]["student"] = upstream
        config["modules"]["teacher"] = self._get_teacher(upstream)

        super().__init__(
            modules=config["modules"],
            opt_class=config["opt_class"],
            hparams=config,
            run_opts={"device": self.args.device},
            checkpointer=config["checkpointer"],
        )

        self.prepare_datasets()
        self.target_ema_updater = EMA(alpha=self.hparams.alpha_ema)

        self.checkpointer.add_recoverable("student", self.modules.student)
        self.checkpointer.add_recoverable("teacher", self.modules.teacher)


    @torch.no_grad()
    def _get_teacher(self, upstream):
        return copy.deepcopy(upstream)

    @torch.no_grad()
    def update_moving_average(self):
        for student_params, teacher_params in zip(
           self.modules.student.parameters(),
           self.modules.teacher.parameters()
        ):
            old_weight, up_weight = teacher_params.data, student_params.data
            teacher_params.data = self.target_ema_updater.update_average(old_weight, up_weight)
        
        for student_params, teacher_params in zip(
           self.modules.projection_student.parameters(),
           self.modules.projection_teacher.parameters()
        ):
            old_weight, up_weight = teacher_params.data, student_params.data
            teacher_params.data = self.target_ema_updater.update_average(old_weight, up_weight)

    def normalize_signal(self, sig):
        return (sig - sig.mean(dim=0)) / (sig.std(dim=0) + 1e-5)

    def student_forward(self, sig):
        embd_student = self.modules.student(sig.squeeze())["hidden_states"][-1]
        embd_student = embd_student.transpose(1, 2)
        projection_student = self.modules.projection_student(embd_student)

        return self.modules.prediction_student(projection_student)

    @torch.no_grad()
    def teacher_forward(self, sig):
        embd_teacher = self.modules.teacher(sig.squeeze())["hidden_states"][-1]
        embd_teacher = embd_teacher.transpose(1, 2)
        return self.modules.projection_teacher(embd_teacher).detach_()

    def compute_forward(self, batch, stage):
        # Get the data, prenorm + augment
        sig, lens = batch["bee_sig"]
        sig = sig.to(self.device)
        sig = self.normalize_signal(sig)
        sig, sig_prime = self.hparams.sig_transform(sig)

        # Forward pass
        pred_student = self.student_forward(sig)
        proj_teacher_prime = self.teacher_forward(sig_prime)

        pred_student_prime = self.student_forward(sig_prime)
        proj_teacher = self.teacher_forward(sig)

        return (pred_student, proj_teacher_prime), (pred_student_prime, proj_teacher)

    def compute_objectives(self, predictions, batch, stage):
        (pred_student, proj_teacher_prime), (pred_student_prime, proj_teacher) = predictions
        loss_a = self.hparams.loss(pred_student, proj_teacher_prime)
        loss_b = self.hparams.loss(pred_student_prime, proj_teacher)

        if (stage != sb.Stage.TRAIN):
            self.loss_metric.append(
                ids=batch.id,
                z=pred_student,
                z_prime=proj_teacher_prime
            )

            self.loss_metric.append(
                ids=batch.id,
                z=pred_student_prime,
                z_prime=proj_teacher
            )

        return (loss_a + loss_b).mean()

    def on_stage_start(self, stage, epoch=None):
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=self.hparams.loss
        )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        if stage != sb.Stage.TRAIN:
            # Summarize the statistics from the stage for record-keeping.
            loss_metric = self.loss_metric.summarize()
            stats = {"byol_loss": loss_metric["average"]}

        # At the end of validation, we can write stats, checkpoints and update LR.
        if (stage != sb.Stage.TRAIN):
            current_lr, next_lr = self.hparams.lr_scheduler(stage_loss)
            schedulers.update_learning_rate(self.optimizer, next_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch": epoch, "LR": current_lr},
                train_stats={"byol_loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            # unless they have the current best task1_metric
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["byol_loss"])

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        self.update_moving_average()

    def pretrain_upstream(self):
        print("[PretrainBrain] - Starting the training process")
        self.fit(
            epoch_counter=self.hparams.epoch_counter,
            train_set=self.datasets["train"],
            valid_set=self.datasets["valid"],
            train_loader_kwargs=self.hparams.dataloader_options,
            valid_loader_kwargs=self.hparams.dataloader_options,
        )

    def prepare_datasets(self):
        prepare_nectar(
            data_folder=self.config["data_root"],
            save_folder=self.config["annotation_folder"],
            skip_prep=False
        )
        self.datasets = dataio_prep(self.config)
