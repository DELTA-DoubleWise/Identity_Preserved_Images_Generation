import torch
from torch import nn
import numpy as np
from typing import List
from functools import partial
import torch.nn.functional as F
import torch.nn.init as init
from model_utils import StyleVectorizer


def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, return_length=True,
                               return_overflowing_tokens=False, return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    # assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens


def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))


def masked_diffusion_loss(face_mask, hair_mask, noise_indices, model_pred, target, l_hair_diff_lambda=0.1):
    fg_loss_noise = F.mse_loss(face_mask[noise_indices] * model_pred[noise_indices].float(),
                               face_mask[noise_indices] * target[noise_indices].float(), reduction="mean")
    hair_loss_noise = F.mse_loss(hair_mask[noise_indices] * model_pred[noise_indices].float(),
                                 hair_mask[noise_indices] * target[noise_indices].float(), reduction="mean")
    noise_loss = fg_loss_noise + hair_loss_noise * l_hair_diff_lambda

    return noise_loss


class IDPreservedGenerativeModel(nn.Module):
    def __init__(self, tokenizer,
                 vision_model,
                 text_model,
                 unet,
                 vae,
                 noise_scheduler,
                 num_embeds_per_token: int = 2,
                 mlp_depth: int = 2,
                 token_dim: int = 1024,
                 vit_out_dim: int = 512):
        super(IDPreservedGenerativeModel, self).__init__()

        self.vision_encoder = vision_model
        self.text_encoder = text_model
        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler

        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.face_projection_layer = StyleVectorizer(vit_out_dim, token_dim * self.num_es,
                                                     depth=mlp_depth, lr_mul=0.1)

    def forward(
            self,
            tokenized_text,
            pixel_values
    ):
        # Convert images to latent space
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        # Sample a random timestep for each image

        timestep = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
        timestep = timestep.long()

        # TODO: Get the text embedding for conditioning
        encoder_hidden_states = None

        # forward diffusion
        noisy_latents, sqrt_alpha_prod, sqrt_one_minus_alpha_prod = self.noise_scheduler.add_noise(latents, noise, timestep)

        model_pred = self.unet(noisy_latents.float(), timestep, encoder_hidden_states).sample

        return model_pred


