import torch
from torch import nn
import numpy as np
from typing import List
from functools import partial
import torch.nn.functional as F
import torch.nn.init as init
from diffusers import StableDiffusionPipeline
from huggingface_hub.utils import validate_hf_hub_args
from model_utils import StyleVectorizer, get_rep_pos
from typing import Any, Callable, Dict, List, Optional, Union
import PIL.Image


PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.FloatTensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.FloatTensor],
]


def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, return_length=True,
                               return_overflowing_tokens=False, return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    # assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens


# def get_embedding_for_clip_token(embedder, token):
#     return embedder(token.unsqueeze(0))


def masked_diffusion_loss(face_mask, hair_mask, model_pred, target, l_hair_diff_lambda=0.1):
    fg_loss_noise = F.mse_loss(face_mask * model_pred.float(),
                               face_mask * target.float(), reduction="mean")
    hair_loss_noise = F.mse_loss(hair_mask * model_pred.float(),
                                 hair_mask * target.float(), reduction="mean")
    noise_loss = fg_loss_noise + hair_loss_noise * l_hair_diff_lambda

    return noise_loss


class IDPreservedGenerativeModel(StableDiffusionPipeline):
    @validate_hf_hub_args
    def load_adaptor(
            self,
            vit_out_dim,
            token_dim,
            mlp_depth,
            pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
            **kwargs
    ):
        self.get_token_for_string = partial(get_clip_token_for_string, self.tokenizer)
        self.placeholder_token = self.get_token_for_string("*")[0][1]

        self.face_projection_layer = StyleVectorizer(vit_out_dim, token_dim * self.num_es,
                                                     depth=mlp_depth, lr_mul=0.1)

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            pixel_values: Optional[torch.Tensor] = None,
            image_embedding: Optional[torch.Tensor] = None,
            face_mask: Optional[torch.Tensor] = None,
            hair_mask: Optional[torch.Tensor] = None
    ):
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]

        # Sample a random timestep for each image
        timestep = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,),
                                 device=latents.device)
        timestep = timestep.long()

        encoder_hidden_states = self._encode_prompt(prompt)
        residual_embedding = self.face_projection_layer(image_embedding).view(batch_size, self.num_es,self.token_dim)
        placeholder_pos = get_rep_pos(encoder_hidden_states, [self.placeholder_token])
        placeholder_pos = np.array(placeholder_pos)

        if len(placeholder_pos) != 0:
            encoder_hidden_states[placeholder_pos[:, 0], placeholder_pos[:, 1]] = residual_embedding[:, 0]
            encoder_hidden_states[placeholder_pos[:, 0], placeholder_pos[:, 1] + 1] = residual_embedding[:, 1]

        # forward diffusion
        noisy_latents, sqrt_alpha_prod, sqrt_one_minus_alpha_prod = self.noise_scheduler.add_noise(latents, noise,
                                                                                                   timestep)

        model_pred = self.unet(noisy_latents.float(), timestep, encoder_hidden_states).sample

        loss = masked_diffusion_loss(face_mask, hair_mask, model_pred, noise)

        return model_pred, loss
