"""Masked Modeling Duo (M2D) Runtime class/functions.
"""

import sys
sys.path.append('..')  # workaround for using heareval with `pip install -e .`

import logging
from pathlib import Path

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange
import nnAudio.features
import re

from . import models_mae
from .timm_layers_pos_embed import resample_abs_pos_embed 


class Config:
    weight_file = ''
    feature_d = 768 * 5
    norm_type = all
    pooling_type = 'mean'

    model = ''
    input_size = [80, 208]
    patch_size = [16, 16]
    cls_token = False
    training_mask = 0.0
    flat_features = False
    encoder_only = True  # For using in fine-tuning
    dur_frames = None    # None for no desired number of frames

    # FFT parameters.
    sample_rate = 16000
    n_fft = 400
    window_size = 400
    hop_size = 160
    n_mels = 80
    f_min = 50
    f_max = 8000
    window = 'hanning'


def parse_sizes_by_name(name):
    model_cls = name.split('-')[0]
    params = name.split('-')[1]
    input_str, patch_str = params.split('p')[:2]
    input_size = [int(a) for a in input_str.split('x')]
    patch_size = [int(a) for a in patch_str.split('x')]
    return input_size, patch_size, model_cls


def drop_non_model_weights(model, checkpoint, filename):
    model_keys = [n for n, p in model.named_parameters()]
    new_ckpt = {}
    for k in checkpoint:
        if k not in model_keys: continue
        new_ckpt[k] = checkpoint[k]
    n_org = len(checkpoint.keys())
    n_cur = len(new_ckpt.keys())
    print(f' using {n_cur} parameters, while dropped {n_org - n_cur} out of {n_org} parameters from {Path(filename).parent/Path(filename).name}'
          if n_org > n_cur else f' using {n_cur} parameters from {Path(filename).parent/Path(filename).name}')
    return new_ckpt

def get_model(args, weight_file, encoder_only, dur_frames):
    # determine model parameters for creation
    try:
        args.input_size, args.patch_size, args.model = parse_sizes_by_name(Path(weight_file).parent.name)
    except:
        args.input_size, args.patch_size, args.model = parse_sizes_by_name(Path(weight_file).stem)
    if dur_frames is not None:
        org_input_size = args.input_size.copy()
        args.input_size[1] = dur_frames

    if encoder_only:
        args.model = args.model + '_encoder_only'
    if Path(weight_file).name.endswith('random'):
        checkpoint = None
        dec_blocks_nums = [4 - 1] # fixed for random init.
        print(' **CAUTION: Random Weights**')
        logging.info(' **CAUTION: Random Weights**')
    else:
        checkpoint = torch.load(weight_file, map_location='cpu')
        checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
        # determine # of decoder blocks
        dec_blocks_nums = [int(k.split('.')[1]) for k in checkpoint.keys() if k.startswith('decoder_blocks.')]
    args.decoder_depth = max(dec_blocks_nums) + 1

    logging.info(f'Creating model: {args.model}(input={args.input_size}, patch={args.patch_size}, decoder_depth={args.decoder_depth})')
    model = models_mae.__dict__[args.model](img_size=args.input_size, patch_size=args.patch_size, decoder_depth=args.decoder_depth)

    # set feature_d
    args.flat_features = True if args.training_mask > 0.0 else args.flat_features
    n_stack_feature = 1 if args.flat_features else (args.input_size[0] // args.patch_size[0])
    d = model.pos_embed.shape[-1]
    args.feature_d = d * n_stack_feature

    # load weights
    if checkpoint:
        # interpolate pos_embed
        if dur_frames is not None:
            org_grid_size = [org_input_size[0] // args.patch_size[0], org_input_size[1] // args.patch_size[1]]
            new_grid_size = [args.input_size[0] // args.patch_size[0], args.input_size[1] // args.patch_size[1]]
            if org_grid_size[1] < new_grid_size[1]:
                checkpoint['pos_embed'] = resample_abs_pos_embed(checkpoint['pos_embed'], old_size=org_grid_size, new_size=new_grid_size)
                print(' resampled pos_embed from', org_grid_size, 'to', new_grid_size, '- new pos_embed shape is', checkpoint['pos_embed'].shape)
            elif org_grid_size[1] > new_grid_size[1]:
                posemb = checkpoint['pos_embed']
                _, _, D = posemb.shape
                posemb_prefix, posemb = posemb[:, :1], posemb[:, 1:]
                posemb = posemb.reshape(1, org_grid_size[0], org_grid_size[1], D)
                posemb = posemb[:, :, :new_grid_size[1], :].reshape(1, new_grid_size[0]*new_grid_size[1], D)
                checkpoint['pos_embed'] = torch.cat([posemb_prefix, posemb], dim=1)
                print(' trimmed pos_embed from', org_grid_size, 'to', new_grid_size, '- new pos_embed shape is', checkpoint['pos_embed'].shape)

        # remove non-model parameters (i.e. for using encoder only model)
        checkpoint = drop_non_model_weights(model, checkpoint, weight_file)
        msg = model.load_state_dict(checkpoint)
        print(msg)
        logging.info(msg)

    model.eval()
    return model


def get_to_melspec(cfg):
    to_spec = nnAudio.features.MelSpectrogram(
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.window_size,
        hop_length=cfg.hop_size,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.f_max,
        center=True,
        power=2,
        verbose=False,
    )
    logging.info(f'Runtime MelSpectrogram({cfg.sample_rate}, {cfg.n_fft}, {cfg.window_size}, {cfg.hop_size}, '
                 + f'{cfg.n_mels}, {cfg.f_min}, {cfg.f_max}):')
    logging.info(to_spec)
    return to_spec


def get_timestamps(cfg, batch_audio, x):  # Returns timestamps in milliseconds.
    audio_len = len(batch_audio[0])
    sec = audio_len / cfg.sample_rate
    x_len = len(x[0])
    step = sec / x_len * 1000 # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(len(batch_audio), 1)
    return ts


class RuntimeM2D(nn.Module):
    def __init__(self, cfg=Config(), weight_file=None, training_mask=0.0, encoder_only=None, dur_frames=None, num_classes=None, head_norm='layernorm'):
        super().__init__()
        cfg.weight_file = weight_file or cfg.weight_file
        cfg.training_mask = training_mask if training_mask > 0.0 else cfg.training_mask
        self.cfg = cfg
        cfg.encoder_only = cfg.encoder_only if encoder_only is None else encoder_only
        cfg.dur_frames = cfg.dur_frames if dur_frames is None else dur_frames
        self.backbone = get_model(cfg, cfg.weight_file, cfg.encoder_only, cfg.dur_frames)
        # runtime masking -> structured mask for audio
        if self.is_training_mask():
            self.backbone.set_random_structured_mask()

        logging.info(str(cfg))
        logging.info(f'Model input size: {cfg.input_size}')
        logging.info(f'Using weights: {cfg.weight_file}')
        logging.info(f'[CLS] token?: {cfg.cls_token}')
        logging.info(f'training_mask: {cfg.training_mask}')
        logging.info(f'flat_features: {cfg.flat_features}')

        self.to_spec = get_to_melspec(cfg)

        self.sample_rate = cfg.sample_rate

        if num_classes is not None:
            assert head_norm in ['layernorm', 'batchnorm']
            self.head_norm = torch.nn.LayerNorm(cfg.feature_d) if head_norm == 'layernorm' else torch.nn.BatchNorm1d(cfg.feature_d, affine=False)
            self.head = torch.nn.Linear(cfg.feature_d, num_classes)
            trunc_normal_(self.head.weight, std=2e-5)

    def forward(self, lms):
        assert hasattr(self, 'head'), 'Set the option num_classes with your desired number of classes, such as 527 for AudioSet.'
        x = self.encode_lms(lms)  # B, T, D
        x = x.mean(1)  # B, D
        x = self.head_norm(x) if isinstance(self.head_norm, torch.nn.LayerNorm) else self.head_norm(x.unsqueeze(-1)).squeeze(-1)
        x = self.head(x)
        return x

    def is_training_mask(self):
        return self.cfg.training_mask > 0.0

    def to_feature(self, batch_audio):
        # raw -> spectrogram, and normalize
        x = self.to_spec(batch_audio)
        x = (x + torch.finfo().eps).log()
        x = x.unsqueeze(1)
        return x

    def normalize_batch(self, x, return_stats=False):
        mu, sigma = x.mean(), x.std()
        x = (x - mu) / sigma
        if return_stats:
            return x, (mu, sigma)
        return x

    def to_normalized_spec(self, batch_audio, return_stats=False):
        # raw -> spectrogram
        x = self.to_feature(batch_audio)
        # normalize among batch samples
        x = self.normalize_batch(x, return_stats=return_stats)
        return x

    def encode_lms(self, lms, return_layers=False):
        x = lms

        patch_fbins = self.backbone.grid_size()[0]
        unit_frames = self.cfg.input_size[1]
        patch_frames = self.backbone.patch_size()[1]
        embed_d = self.backbone.patch_embed.proj.out_channels
        pad_frames = (patch_frames - x.shape[-1] % patch_frames) % patch_frames
        if pad_frames > 0:
            x = torch.nn.functional.pad(x, (0, pad_frames))
        chunks = (x.shape[-1] + unit_frames - 1) // unit_frames

        embeddings = []
        if self.cfg.flat_features:
            # flatten all patch embeddings
            mask_ratio = self.cfg.training_mask if self.training else 0.0
            for i in range(chunks):
                emb, *_ = self.backbone.forward_encoder(x[..., i*unit_frames:(i+1)*unit_frames], mask_ratio=mask_ratio, return_layers=return_layers, adjust_short=True)
                cls_token, emb = emb[..., :1, :], emb[..., 1:, :]
                if self.cfg.cls_token:
                    # prepend cls token to all frame features.
                    # in:
                    #   cls_token.shape -> [B, 1, D]
                    #   emb.shape -> [B, T*F, D]
                    # out:
                    #   emb.shape -> [B, 1 + T*F, D]
                    emb = torch.cat([cls_token, emb], axis=-1)
                embeddings.append(emb)
        else:
            # stack embeddings along time frame
            for i in range(chunks):
                emb, *_ = self.backbone.forward_encoder(x[..., i*unit_frames:(i+1)*unit_frames], mask_ratio=0., return_layers=return_layers, adjust_short=True)
                cls_token, emb = emb[..., :1, :], emb[..., 1:, :]
                if len(emb.shape) > 3:
                    emb = rearrange(emb, 'L b (f t) d -> L b t (f d)', f=patch_fbins, d=embed_d)  # Layer-wise embeddings
                else:
                    emb = rearrange(emb, 'b (f t) d -> b t (f d)', f=patch_fbins, d=embed_d)

                if self.cfg.cls_token:
                    # prepend cls token to all frame features.
                    #  cat([L, B, 1, D].repeat(1, T, 1), [L, B, T, F*D]) -> [L, B, T, (1 + F)*D] or
                    #  cat([B, 1, D].repeat(1, T, 1), [B, T, F*D]) -> [B, T, (1 + F)*D]
                    emb = torch.cat([cls_token.repeat(*([1]*(len(emb.shape) - 2)), emb.shape[-2], 1), emb], axis=-1)
                embeddings.append(emb)
        # concatenate chunks in the time axis
        x = torch.cat(embeddings, axis=-2)
        return x if len(x.shape) == 3 else [x_ for x_ in x]

    def encode(self, batch_audio):
        x = self.to_normalized_spec(batch_audio)
        return self.encode_lms(x)

    def get_scene_embeddings(self, audio):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        Returns:
            embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
        """
        x = self.encode(audio)
        x = torch.mean(x, dim=1)
        return x

    def get_timestamp_embeddings(self, audio):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        Returns:
            embedding: A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
            timestamps: A float32 Tensor with shape (`n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
        """
        x = self.encode(audio)
        ts = get_timestamps(self.cfg, audio, x)
        # print(audio.shape, x.shape, ts.shape)
        return x, ts

    def reconstruct(self, lms, mask_ratio, start_frame=0):
        """A helper function to get reconstruction results.
        Use `lms_to_wav` if you may also want to convert the reconstruction results to wavs.
        **Note** this does *not* process the entire LMS frames but rather crops them from the start_frame with the duration of the model's unit frame.
        """

        # trim frames
        unit_frames = self.backbone.patch_embed.img_size[1]
        last_frame = start_frame + unit_frames
        lms_cropped = lms[..., start_frame:last_frame]
        # raw reconstruction
        with torch.no_grad():
            loss, recons, errormap, mask = self.backbone.forward_viz(lms_cropped, mask_ratio)

        return loss, lms_cropped, recons, errormap, mask

    def decode_to_lms(self, lms_all):
        """Decode the embeddings into LMS.
        Note: To be very strict, we cannot guarantee that the decoder can reconstruct visible patch embeddings to the original LMS space
        because the training does not calculate the loss on the reconstruction result of the visible patches. Since the loss is only calculated on the masked tokens,
        the decoder learns to predict the original input patches of the masked tokens using the visible patch tokens.
        """
        ids_restore = torch.tensor(list(range(lms_all.shape[-2] - 1))).repeat(lms_all.shape[0], 1)
        with torch.no_grad():
            preds = self.backbone.forward_decoder(lms_all, ids_restore)
        decoded = self.backbone.unpatchify(preds)
        return decoded

    def lms_to_wav(self, single_lms, norm_stats, sr=16000, n_fft=400, hop_length=160, win_length=400):
        """A helper function to revert an LMS into an audio waveform.
        CAUTION: Be sure to use the normalization statistics you used to normalize the LMS.
        """

        mu, sigma = norm_stats
        M = (single_lms*sigma + mu).exp().numpy()
        wav = librosa.feature.inverse.mel_to_audio(M, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        # display(Audio(wav, rate=sr))
        return wav
