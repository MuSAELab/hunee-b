# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from functools import partial

import torch
import torch.nn as nn
import numpy as np

import timm
from timm.models.vision_transformer import Block

from pos_embed import get_2d_sincos_pos_embed


def expand_size(sz):
    if isinstance(sz, int):
        return [sz, sz]
    return sz


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding -- borrowed from https://pypi.org/project/timm/0.4.12/
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = expand_size(img_size)
        patch_size = expand_size(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def random_unstructured_mask(shape, mask_ratio, device):
    B, F, T = shape # Batch, Freq bins, and Time frames; equivalent to Batch, Height, and Width for the image.
    L = F * T
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(B, L, device=device)  # noise in [0, 1]
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    return ids_shuffle, len_keep


def random_structured_mask(shape, mask_ratio, device):
    """Random structured masking for training in audio tasks."""
    B, F, T = shape

    # We want true random freq/time masking but need to make the number of masks consistent among samples.
    # We impose a constraint that the number of freq/time masks be consistent across samples while leaving it open where we mask.
    NF = int(F * (mask_ratio + 1./F) * np.random.rand())
    NF = min(F - 1, NF) # prevent to mask all freq. bins.
    mask_ratio = max(mask_ratio + (.5/T) - (NF/F), 0.)
    NT = int(T*mask_ratio)

    # Make mask for each batch sample.
    mask = torch.zeros((B, F, T), dtype=torch.int, device=device)
    for b in range(B):
        mask[b, torch.randperm(F)[:NF]] = 1
    for b in range(B):
        mask[b, :, torch.randperm(T)[:NT]] = 1

    ids_shuffle = torch.argsort(mask.view(B, -1), descending=True)
    len_keep = (mask[0] == 0).sum()
    # print(len_keep, mask[:2])
    return ids_shuffle, len_keep


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.in_chans = in_chans
        img_size, patch_size = expand_size(img_size), expand_size(patch_size)

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.img_patch_dim(), bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        print(f'{self.__class__.__name__}(in_chans={self.in_chans}, patch size={self.patch_size()}, grid_size={self.grid_size()},\n'
              f'  embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}, decoder_embed_dim={decoder_embed_dim},\n'
              f'  decoder_depth={decoder_depth}, decoder_num_heads={decoder_num_heads}, mlp_ratio={mlp_ratio},\n'
              f'  norm_pix_loss={norm_pix_loss})')

        self.initialize_weights()

        self._random_mask_fn = random_unstructured_mask

    def set_random_structured_mask(self):
        print('using random_structured_mask().')
        self._random_mask_fn = random_structured_mask

    def patch_size(self):
        return self.patch_embed.proj.kernel_size

    def grid_size(self):
        # This fails with timm 0.4.5 -> return self.patch_embed.grid_size
        # Workaround for avoid compatibility issue
        img_size = np.array(self.patch_embed.img_size)
        patch_size = np.array(self.patch_embed.patch_size)
        grid_size = img_size // patch_size
        return grid_size

    def img_patch_dim(self):
        patch_size = self.patch_size()
        return patch_size[0] * patch_size[1] * self.in_chans

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size(), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size(), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size[0]*patch_size[0]*in_chans)
        """
        ph, pw = self.patch_size()
        h, w = self.grid_size()
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, ph, w, pw))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, self.img_patch_dim()))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size[0]*patch_size[0]*in_chans)
        imgs: (N, C, H, W)
        """
        ph, pw = self.patch_size()
        h, w = self.grid_size()
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * ph, w * pw))
        return imgs

    def random_masking(self, x, mask_ratio, adjust_short=False):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim

        if isinstance(mask_ratio, (torch.Tensor, np.ndarray)):
            # Prefixed mask. `mask` shall be 2x sized.
            mask = mask_ratio.clone().detach()
            #ids_shuffle = torch.where(mask.reshape(N, -1) == 0)[1].reshape(N, -1)
            ids_shuffle = torch.argsort(mask.reshape(N, -1), dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            len_keep = (mask[0] == 0).sum() // 2
        elif isinstance(mask_ratio, (list, tuple)):
            # Prefixed ids_restore & len_keep.
            ids_restore = mask_ratio[0]
            ids_shuffle = torch.argsort(ids_restore, dim=1)
            len_keep = mask_ratio[1]
        elif mask_ratio == 0:
            # No mask
            mask = torch.zeros([N, L], device=x.device)
            ids_restore = torch.tensor(list(range(L))).to(torch.int)
            return x, mask, ids_restore
        else:
            # Random mask
            HorF, WorT = self.grid_size()
            if adjust_short and L < HorF * WorT:
                # audio: shorten pos_embed for a short input
                WorT = L // HorF
            ids_shuffle, len_keep = self._random_mask_fn((N, HorF, WorT), mask_ratio, x.device)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, return_layers=False, adjust_short=False):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        pos_embed = self.pos_embed[:, 1:, :]
        if adjust_short and x.shape[1] < pos_embed.shape[1]:
            # audio: shorten pos_embed for a short input
            dims = pos_embed.shape[-1]
            fbins = self.grid_size()[0]
            frames = x.shape[1] // fbins
            pos_embed = pos_embed.reshape(1, fbins, -1, dims)[:, :, :frames, :].reshape(1, fbins*frames, dims)
        x = x + pos_embed

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, adjust_short=adjust_short)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        layers = []
        for blk in self.blocks:
            x = blk(x)
            if return_layers: layers.append(x)
        x = self.norm(x)
        if return_layers:
            layers.pop() # replace the last feature with the normalized one.
            layers.append(x)

        if return_layers:
            return torch.stack(layers), mask, ids_restore
        return x, mask, ids_restore

    def drop_cls_token(self, latent):
        # remove cls token [B, 1+H*W: D] -> [B, H*W, D]
        # L = latent.shape[-2]
        # assert L == 1 + self.grid_size()[0]*self.grid_size()[1], f'Already no class token...? {L} vs {self.grid_size()}'
        return  latent[:, 1:, :]

    def get_cls_token(self, latent):
        # return cls token only [B, 1+H*W: D] -> [B, 1, D]
        # L = latent.shape[-2]
        # assert L == 1 + self.grid_size()[0]*self.grid_size()[1], f'No class token...? {L} vs {self.grid_size()}'
        return  latent[:, :1, :]

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, ph*pw*C]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        # if torch.isnan(loss).any():
        #     print('loss contains nan(s), which is replaced with 0...')
        #     loss = torch.nan_to_num(loss)
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def forward_viz(self, imgs, mask_ratio=0.75):
        loss, pred, mask = self.forward(imgs, mask_ratio)
        # recons_as_is = self.unpatchify(pred)
        # overwrite visible patches with original image.
        pred_org_on_mask = pred.clone()
        visible = (mask == 0.)
        pred_org_on_mask[visible] = self.patchify(imgs)[visible]
        recons = self.unpatchify(pred_org_on_mask)
        errormap = ((recons - imgs) ** 2).sqrt()
        return loss, recons, errormap, mask.reshape(mask.shape[0], *self.grid_size())


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


def mae_vit_base(patch_size=16, decoder_depth=8, in_chans=1, **kwargs):
    model = MaskedAutoencoderViT(
        in_chans=in_chans, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=decoder_depth, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def msm_mae_vit_base(patch_size=16, decoder_depth=8, in_chans=1, **kwargs):
    print(f'MSM-MAE **FORCED DEC DEPTH AS 4 (Your decoder_depth={decoder_depth} is ignored)**')
    model = MaskedAutoencoderViT(
        in_chans=in_chans, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# Masked Modeling Duo (M2D)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def ema_model_weight(decay, old_model, new_model):
    def ema(decay, old, new):
        return old * decay + (1 - decay) * new

    for new_params, old_params in zip(new_model.parameters(), old_model.parameters()):
        old_weight, new_weight = old_params.data, new_params.data
        old_params.data = ema(decay, old_weight, new_weight)


class M2DViT(MaskedAutoencoderViT):
    """ Masked Modeling Duo (M2D) implementation based on the MAE.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 loss_type='norm_mse', target_layers=None, **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                 embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                 decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
                 mlp_ratio=mlp_ratio, norm_layer=norm_layer, norm_pix_loss=norm_pix_loss)
        self.loss_type = loss_type
        self.target_layers = target_layers
        print(f'+ loss_type={loss_type}, target_layers={target_layers}')
        if len(kwargs.keys()) > 0:
            print(' CAUTION: You set unknown arguments ->', kwargs)

        # --------------------------------------------------------------------------
        # Target encoder specifics
        self.target_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.target_norm = norm_layer(embed_dim)
        set_requires_grad(self.target_blocks, False)
        set_requires_grad(self.target_norm, False)
        self.target_blocks.apply(self._init_weights)
        self.target_norm.apply(self._init_weights)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Decoder specifics
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True) # predict target embeddings
        self.decoder_pred.apply(self._init_weights)
        # --------------------------------------------------------------------------

    def update_target_network(self, ema_decay):
        ema_model_weight(ema_decay, self.target_blocks, self.blocks)
        ema_model_weight(ema_decay, self.target_norm, self.norm)

    def random_masking(self, x, mask_ratio, adjust_short=False):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim

        if isinstance(mask_ratio, (torch.Tensor, np.ndarray)):
            # Prefixed mask. `mask` shall be 2x sized.
            mask = mask_ratio.clone().detach()
            #ids_shuffle = torch.where(mask.reshape(N, -1) == 0)[1].reshape(N, -1)
            ids_shuffle = torch.argsort(mask.reshape(N, -1), dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            len_keep = (mask[0] == 0).sum() // 2
        elif isinstance(mask_ratio, (list, tuple)):
            # Prefixed ids_restore & len_keep.
            ids_restore = mask_ratio[0]
            ids_shuffle = torch.argsort(ids_restore, dim=1)
            len_keep = mask_ratio[1]
        elif mask_ratio == 0:
            # No mask
            mask = torch.zeros([N, L], device=x.device)
            ids_restore = torch.tensor(list(range(L))).to(torch.int)
            return x, None, mask, ids_restore
        else:
            # Random mask
            HorF, WorT = self.grid_size()
            if adjust_short and L < HorF * WorT:
                # audio: shorten pos_embed for a short input
                WorT = L // HorF
            ids_shuffle, len_keep = self._random_mask_fn((N, HorF, WorT), mask_ratio, x.device)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the visible patch indexes
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # keep the rest
        ids_keep = ids_shuffle[:, len_keep:]
        x_masked2 = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, x_masked2, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, return_layers=False, blocks=None, norm=None, adjust_short=False):
        blocks, norm = blocks or self.blocks, norm or self.norm

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        pos_embed = self.pos_embed[:, 1:, :]
        if adjust_short and x.shape[1] < pos_embed.shape[1]:
            # audio: shorten pos_embed for a short input
            dims = pos_embed.shape[-1]
            fbins = self.grid_size()[0]
            frames = x.shape[1] // fbins
            pos_embed = pos_embed.reshape(1, fbins, -1, dims)[:, :, :frames, :].reshape(1, fbins*frames, dims)
        x = x + pos_embed

        # masking: length -> length * mask_ratio; TODO fix comment
        x, x_targ, mask, ids_restore = self.random_masking(x, mask_ratio, adjust_short=adjust_short)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        layers = []
        for blk in blocks:
            x = blk(x)
            if return_layers: layers.append(x)
        x = norm(x)
        if return_layers:
            layers.pop() # replace the last feature with the normalized one.
            layers.append(x)

        if return_layers:
            return torch.stack(layers), x_targ, mask, ids_restore
        return x, x_targ, mask, ids_restore

    def forward_decoder(self, x, ids_restore, keep_cls=False, also_pred_asis=False):
        len_keep = x.shape[1] - 1 # tokens - cls

        # embed tokens
        x = self.decoder_embed(x)
        D = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        y = self.drop_cls_token(x)
        y_pred_asis = y
        # re-shuffle, and keep prediction only
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        y = torch.gather(y, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, y.shape[-1]))
        y = y[:, len_keep:] # prediction only

        # append cls if needed
        if keep_cls:
            y = torch.cat([x[:, :1, :], y], dim=1)
        if also_pred_asis:
            return y, y_pred_asis
        return y

    def forward_target_encoder(self, x_targ, drop_cls=True):
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_targ.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x_targ), dim=1)

        # apply Transformer blocks
        xs = []
        for l, blk in enumerate(self.target_blocks):
            x = blk(x)
            if self.target_layers and l in self.target_layers:
                xs.append(x)
        if xs:
            x = torch.stack(xs).mean(0)
        x = self.target_norm(x)

        # remove cls token
        if drop_cls:
            x = self.drop_cls_token(x)

        return x

    def forward_loss(self, target, pred, norm_pix_loss, loss_type):
        """
        target: [N, targL, D]
        pred: [N, targL, D]
        """

        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        if loss_type == 'mse':
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per predicted patch embedding
        elif loss_type == 'norm_mse':
            target = torch.nn.functional.normalize(target, dim=-1, p=2)
            pred = torch.nn.functional.normalize(pred, dim=-1, p=2)
            loss = target * pred
            loss = 2 - 2 * loss.sum(dim=-1)
        else:
            assert loss_type in ['WE NEED A KNOWN LOSS FN']

        loss = loss.mean()

        return loss

    def forward(self, imgs, mask_ratio=0.7):
        latent, x_targ, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, targL, D]
        with torch.no_grad():
            target = self.forward_target_encoder(x_targ)

        loss = self.forward_loss(target, pred, self.norm_pix_loss, self.loss_type)
        return loss, pred, (ids_restore, mask)

    def forward_viz(self, imgs, mask_ratio=0.7):
        # Visualize the input and mask.
        loss, pred, (ids_restore, mask) = self.forward(imgs, mask_ratio)
        recons, errormap = None, None
        return recons, errormap, mask.reshape(mask.shape[0], *self.grid_size())


def m2d_vit_base_patch16_dec512d8b(**kwargs):  # for image
    model = M2DViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


m2d_vit_base_patch16 = m2d_vit_base_patch16_dec512d8b  # for image


def m2d_vit_base(patch_size=16, decoder_depth=8, in_chans=1, **kwargs):  # for audio
    model = M2DViT(
        in_chans=in_chans, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=decoder_depth, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# M2D variants

class M2D_D2ViT(M2DViT):
    """A data2vec-like M2D variant that feeds all patches to the target network."""

    def random_masking(self, x, mask_ratio):
        """Random masking that returns all patches as the x_target.

        Returns:
            x_masked: Maked patches
            x_target: Target patches = all patches for Data2Vec
            mask: Mask
            ids_restore: indexes for restoration of masked patches
        """
        x_masked, _, mask, ids_restore = super().random_masking(x, mask_ratio)
        return x_masked, x, mask, ids_restore

    def forward(self, imgs, mask_ratio=0.75):
        latent, x_targ, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, targL, D]
        with torch.no_grad():
            # x_targ holds all the input patches
            target = self.forward_target_encoder(x_targ)
            len_keep = latent.shape[1] - 1 # tokens - cls
            ids_shuffle = torch.argsort(ids_restore, dim=1)
            ids_keep = ids_shuffle[:, len_keep:]
            # target to leave masked patch representations only 
            target = torch.gather(target, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, target.shape[-1]))

        loss = self.forward_loss(target, pred, norm_pix_loss=self.norm_pix_loss, loss_type=self.loss_type)
        return loss, pred, ids_restore # mask


def m2d_d2v_vit_base(patch_size=16, decoder_depth=8, in_chans=1, **kwargs):  # for audio
    model = M2D_D2ViT(
        in_chans=in_chans, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=decoder_depth, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def m2d_d2v_vit_base_patch16_dec512d8b(**kwargs):  # for image
    model = M2D_D2ViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


m2d_d2v_vit_base_patch16 = m2d_d2v_vit_base_patch16_dec512d8b  # for image


class M2DEncoderViT(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Our positional encoding is constant
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size(), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed.requires_grad_(False)
        # Replace PatchEmbed to avoid unintended assertion failure. ex) AssertionError: Input image width (102) doesn't match model (608).
        self.patch_embed = PatchEmbed(self.patch_embed.img_size, self.patch_embed.patch_size,
                                      self.patch_embed.proj.in_channels, self.patch_embed.proj.out_channels)
        # We do not use `head` for the M2D encoder only ViT
        del self.head

    def patch_size(self):
        return self.patch_embed.proj.kernel_size

    def grid_size(self):
        # This fails with timm 0.4.5 -> return self.patch_embed.grid_size
        # Workaround for avoid compatibility issue
        img_size = np.array(self.patch_embed.img_size)
        patch_size = np.array(self.patch_embed.patch_size)
        grid_size = img_size // patch_size
        return grid_size

    def set_random_structured_mask(self):
        print('using random_structured_mask().')
        self._random_mask_fn = random_structured_mask

    def random_masking(self, x, mask_ratio, adjust_short=False):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim

        if isinstance(mask_ratio, (torch.Tensor, np.ndarray)):
            # Prefixed mask. `mask` shall be 2x sized.
            mask = mask_ratio.clone().detach()
            #ids_shuffle = torch.where(mask.reshape(N, -1) == 0)[1].reshape(N, -1)
            ids_shuffle = torch.argsort(mask.reshape(N, -1), dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            len_keep = (mask[0] == 0).sum() // 2
        elif isinstance(mask_ratio, (list, tuple)):
            # Prefixed ids_restore & len_keep.
            ids_restore = mask_ratio[0]
            ids_shuffle = torch.argsort(ids_restore, dim=1)
            len_keep = mask_ratio[1]
        elif mask_ratio == 0:
            # No mask
            mask = torch.zeros([N, L], device=x.device)
            ids_restore = torch.tensor(list(range(L))).to(torch.int)
            return x, None, mask, ids_restore
        else:
            # Random mask
            HorF, WorT = self.grid_size()
            if adjust_short and L < HorF * WorT:
                # audio: shorten pos_embed for a short input
                WorT = L // HorF
            ids_shuffle, len_keep = self._random_mask_fn((N, HorF, WorT), mask_ratio, x.device)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the visible patch indexes
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # keep the rest
        ids_keep = ids_shuffle[:, len_keep:]
        x_masked2 = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, x_masked2, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, return_layers=False, blocks=None, norm=None, adjust_short=False):
        blocks, norm = blocks or self.blocks, norm or self.norm

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        pos_embed = self.pos_embed[:, 1:, :]
        if adjust_short and x.shape[1] < pos_embed.shape[1]:
            # audio: shorten pos_embed for a short input
            dims = pos_embed.shape[-1]
            fbins = self.grid_size()[0]
            frames = x.shape[1] // fbins
            pos_embed = pos_embed.reshape(1, fbins, -1, dims)[:, :, :frames, :].reshape(1, fbins*frames, dims)
        x = x + pos_embed

        # masking: length -> length * mask_ratio; TODO fix comment
        x, x_targ, mask, ids_restore = self.random_masking(x, mask_ratio, adjust_short=adjust_short)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        layers = []
        for blk in blocks:
            x = blk(x)
            if return_layers: layers.append(x)
        x = norm(x)
        if return_layers:
            layers.pop() # replace the last feature with the normalized one.
            layers.append(x)

        if return_layers:
            return torch.stack(layers), x_targ, mask, ids_restore
        return x, x_targ, mask, ids_restore


def m2d_vit_base_encoder_only(patch_size=16, decoder_depth=8, in_chans=1, **kwargs):  # for audio, encoder only
    model = M2DEncoderViT(
        in_chans=in_chans, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


msm_mae_vit_base_encoder_only = m2d_vit_base_encoder_only


# Masked Modeling Duo for Speech (M2D-S)

class M2D_S_ViT(M2DViT):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 loss_type='norm_mse', target_layers=None,
                 loss_m2d=0.7, loss_off=0.3, off_emb_dim=768):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                 embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                 decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
                 mlp_ratio=mlp_ratio, norm_layer=norm_layer, norm_pix_loss=norm_pix_loss,
                 loss_type=loss_type, target_layers=target_layers)
        self.loss_m2d = loss_m2d
        self.loss_off = loss_off
        print(f'+ loss_m2d={loss_m2d}, loss_off={loss_off}')

        F = self.grid_size()[0]
        self.offline_predictor = nn.Linear(F * embed_dim, off_emb_dim)

    def forward(self, imgs, target_offline, mask_ratio=0.7):
        # online
        latent, x_targ, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred_online = self.forward_decoder(latent, ids_restore)  # [N, targL, D]
        with torch.no_grad():
            target_online = self.forward_target_encoder(x_targ)
        loss_online = self.forward_loss(target_online, pred_online, self.norm_pix_loss, self.loss_type)

        # offline
        z_online = torch.cat([self.drop_cls_token(latent), pred_online], dim=1)  # gather all the online tokes
        z_online = torch.gather(z_online, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, latent.shape[-1]))  # unshuffle
        B, FT, D = z_online.shape
        F = self.grid_size()[0]
        T = FT // F
        z_online = torch.einsum('bftd->btfd', z_online.reshape(B, F, T, D)).reshape(B, T, F * D)
        pred_offline = self.offline_predictor(z_online)

        B, T2, D = target_offline.shape
        assert T2 == T, f'label length {T2} != patch length'

        loss_offline = self.forward_loss(target_offline, pred_offline, self.norm_pix_loss, self.loss_type)

        loss = self.loss_m2d * loss_online + self.loss_off * loss_offline
        return loss, pred_online, (ids_restore, mask, loss_online, loss_offline)


def m2d_s_vit_base(patch_size=16, decoder_depth=8, in_chans=1, **kwargs): # for audio
    model = M2D_S_ViT(
        in_chans=in_chans, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=decoder_depth, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


m2d_s_vit_base_encoder_only = m2d_vit_base_encoder_only
