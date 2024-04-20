import torch
from torch import nn
import numpy as np
from typing import List
from functools import partial
import torch.nn.functional as F
import torch.nn.init as init
from diffusers import StableDiffusionPipeline
from huggingface_hub.utils import validate_hf_hub_args
from model_utils import StyleVectorizer
from util import get_rep_pos
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
            vit_out_dim: int = 768,
            token_dim: int = 1024,
            mlp_depth: int = 2,
            num_embeds_per_token: int = 2,
            weight_dtype: Optional[torch.dtype] = torch.float16,
            device: Optional[Union[str, torch.device]] = None,
            face_image_embedding: Optional[torch.Tensor] = None,
            # pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
            **kwargs
    ):
        self.get_token_for_string = partial(get_clip_token_for_string, self.tokenizer)
        self.placeholder_token = self.get_token_for_string("*")[0][1]
        self.num_es = num_embeds_per_token
        self.weight_dtype = weight_dtype
        self.token_dim = token_dim
        self.face_image_embedding = face_image_embedding.to(device)

        self.face_projection_layer = StyleVectorizer(vit_out_dim, token_dim * self.num_es, depth=mlp_depth, lr_mul=0.1).to(device)

        # Freeze parameters in the VAE
        for param in self.vae.parameters():
            param.requires_grad = False

        # Freeze parameters in the UNet
        for param in self.unet.parameters():
            param.requires_grad = False

        # Freeze parameters in the text_encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False

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
            hair_mask: Optional[torch.Tensor] = None,
            device: Optional[Union[str, torch.device]] = None,
    ):  
        self.vae = self.vae.to(device)
        self.unet = self.unet.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.face_projection_layer = self.face_projection_layer.to(device)

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]

        # Sample a random timestep for each image
        timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,),
                                 device=latents.device)
        timestep = timestep.long()

        encoder_hidden_states, _ = self.encode_prompt(prompt, device, 1, False, image_embedding = image_embedding)

        # forward diffusion
        # noisy_latents, sqrt_alpha_prod, sqrt_one_minus_alpha_prod = self.scheduler.add_noise(latents, noise, timestep)
        noisy_latents, _, _ = self.scheduler.add_noise(latents, noise, timestep)
        model_pred = self.unet(noisy_latents, timestep, encoder_hidden_states).sample
        loss = masked_diffusion_loss(face_mask, hair_mask, model_pred, noise)

        return model_pred, loss

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        image_embedding: Optional[torch.Tensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        # if lora_scale is not None and isinstance(self, LoraLoaderMixin):
        #     self._lora_scale = lora_scale

        #     # dynamically adjust the LoRA scale
        #     if not USE_PEFT_BACKEND:
        #         adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
        #     else:
        #         scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            seq_length = text_input_ids.shape[-1]
            
            text_embeddings = self.text_encoder.text_model.embeddings.token_embedding(text_input_ids)

            text_img_embeddings = self.face_projection_layer(image_embedding).view(batch_size, self.num_es, self.token_dim)

            placeholder_pos = get_rep_pos(text_input_ids, [self.placeholder_token])
            placeholder_pos = np.array(placeholder_pos)
            if len(placeholder_pos) != 0:
                text_embeddings[placeholder_pos[:, 0], placeholder_pos[:, 1]] = text_img_embeddings[:, 0]
                text_embeddings[placeholder_pos[:, 0], placeholder_pos[:, 1] + 1] = text_img_embeddings[:, 1]

            position_ids = self.text_encoder.text_model.embeddings.position_ids[:, :seq_length]        
            position_embeddings = self.text_encoder.text_model.embeddings.position_embedding(position_ids)
            hidden_states = text_embeddings + position_embeddings

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                # prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = self.text_encoder.text_model.encoder(inputs_embeds=hidden_states, attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                # prompt_embeds = self.text_encoder(
                #     text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                # )
                prompt_embeds = self.text_encoder.text_model.encoder(
                    inputs_embeds=hidden_states, attention_mask=attention_mask, output_hidden_states=True
                    )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            # if isinstance(self, TextualInversionLoaderMixin):
            #     uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
        #     # Retrieve the original scale by scaling back the LoRA layers
        #     unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def save(self, emb_path):
        # save embeddings
        residual_embedding = self.face_projection_layer(self.face_image_embedding).view(self.face_image_embedding.shape[0], self.num_es, self.token_dim)
        torch.save(residual_embedding, emb_path)
