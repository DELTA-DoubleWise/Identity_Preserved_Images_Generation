#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import itertools
import logging
import math
import os
from pathlib import Path

import accelerate
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available


import numpy as np
from omegaconf import OmegaConf
import random
from transformers import ViTModel
from .models.celeb_embeddings import embedding_forward  
from .models.embedding_manager import EmbeddingManagerId_adain
from .datasets_face.face_id import FaceIdDataset
from .utils import text_encoder_forward, set_requires_grad, latents_to_images, latents_to_images_tensor, add_noise_return_paras
import torch.nn as nn
from torch import autograd
import types
from transformers import ViTImageProcessor



logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision=None):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a script for training Cones 2.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/user/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )    
    parser.add_argument(
        "--vit_face_recognition_model_path",
        type=str,
        default="/home/user/.cache/huggingface/hub/vit-base-patch16-224-in21k-face-recognition",
        help=('config to load the train model and dataset'),
    )         
    parser.add_argument(
        "--embedding_manager_config",
        type=str,
        default="datasets_face/identity_space.yaml",
        help=('config to load the train model and dataset'),
    )
    parser.add_argument("--d_reg_every", type=int, default=16,
                                help="interval for applying r1 regularization")    
    parser.add_argument("--r1", type=float, default=1, help="weight of the r1 regularization")
    parser.add_argument(
        "--l_hair_diff_lambda",
        type=float,
        default=0,
        help="Initial learning rate (after the potential warmup period) to use.",
    )                        
    parser.add_argument(
        "--l_norm_lambda",
        type=float,
        default=0,
        help="Initial learning rate (after the potential warmup period) to use.",
    )           
    parser.add_argument(
        "--pretrained_embedding_manager_path",
        type=str,
        default=None,          
        help="pretrained_embedding_manager_path",
    )
    parser.add_argument(
        "--pretrained_embedding_manager_epoch",
        type=str,
        default=500,
        help="pretrained_embedding_manager_epoch",
    )                
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--face_img_path",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cones2-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=451,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via"
            " `--resume_from_checkpoint`. In the case that the checkpoint is better than the final trained model, the"
            " checkpoint can also be used for inference. Using a checkpoint for inference requires separate loading of"
            " the original pipeline and the individual checkpointed model components."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,           
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def encode_prompt(prompt_batch, text_encoder, tokenizer, embedding_manager, is_train=True,
                  face_img_embeddings = None, timesteps = None):
    prompt_embeds_list = []
    captions = []
    proportion_empty_prompts = 0        
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(text_encoder.device)  

    prompt_embeds = text_encoder_forward(text_encoder = text_encoder, 
                                            input_ids = text_input_ids,
                                            output_hidden_states=True,
                                            embedding_manager = embedding_manager,
                                            face_img_embeddings = face_img_embeddings,
                                            timesteps = timesteps)
    return prompt_embeds




def collate_fn(examples):
    input_ids = [example["face_prompt_ids"] for example in examples]
    pixel_values = [example["face_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {"input_ids": input_ids, "pixel_values": pixel_values}
    return batch


class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {"prompt": self.prompt, "index": index}
        return example
    
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:  
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train_img_to_embedding(processed_image_path, pt_file_path):

    print(f"image path: {processed_image_path}")
    print(f"embedding saving path: {pt_file_path}")
    
    pt_directory = "/".join(pt_file_path.rsplit("/", 2)[:-1])
    os.makedirs(pt_directory, exist_ok=True)
    
    pretrained_model_name = "stabilityai/stable-diffusion-2-1-base"
    vit_face_recognition_model_name = "google/vit-base-patch16-224-in21k"
    embedding_manager_config_path = "StableIdentity_model/datasets_face/identity_space.yaml"
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            subfolder="tokenizer",
            use_fast=False,
        )
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")
    noise_scheduler.add_noise = types.MethodType(add_noise_return_paras, noise_scheduler)

    
    text_encoder = text_encoder_cls.from_pretrained(
        pretrained_model_name, subfolder="text_encoder"
    )
    
    text_encoder.text_model.embeddings.forward = embedding_forward.__get__(text_encoder.text_model.embeddings)

    # face recognition encoder 
    vit_face_recognition_model = ViTModel.from_pretrained(vit_face_recognition_model_name)  
    
    # configs
    embedding_manager_config = OmegaConf.load(embedding_manager_config_path)
    Embedding_Manager = EmbeddingManagerId_adain(
            tokenizer,
            text_encoder,
            device = accelerator.device,
            training = True,
            num_embeds_per_token = embedding_manager_config.model.personalization_config.params.num_embeds_per_token,            
            token_dim = embedding_manager_config.model.personalization_config.params.token_dim,
            mlp_depth = embedding_manager_config.model.personalization_config.params.mlp_depth,
            vit_out_dim = embedding_manager_config.model.personalization_config.params.vit_out_dim,
    )
    Embedding_Manager.face_projection_layer.apply(weights_init_normal)

    for param in Embedding_Manager.trainable_projection_parameters():
        param.requires_grad = True

    vae = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name, subfolder="unet"
    )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )


    optimizer_class = torch.optim.AdamW
    gradient_accumulation_steps = 1
    learning_rate = 5e-5
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08
    train_batch_size = 1
    max_train_steps = 451
    num_train_epochs = 1
    lr_scheduler = "constant"
    lr_num_cycles = 1
    lr_power = 1.0
    l_hair_diff_lambda = 0.1
    resolution = 512
    max_grad_norm = 1.0
        
    projection_params_to_optimize = Embedding_Manager.trainable_projection_parameters()
    optimizer_projection = optimizer_class(
        projection_params_to_optimize,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = FaceIdDataset(
        face_img_path=processed_image_path,
        image_size=resolution,
        vit_path = vit_face_recognition_model_name
    )
    print("dataset_length", train_dataset._length)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=accelerator.num_processes,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_proj = get_scheduler(
        lr_scheduler,
        optimizer=optimizer_projection,
        num_warmup_steps=0,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )


    Embedding_Manager, optimizer_projection, train_dataloader, lr_scheduler_proj = accelerator.prepare(
        Embedding_Manager, optimizer_projection, train_dataloader, lr_scheduler_proj
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32

    # Move vae and unet to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vit_face_recognition_model.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    Embedding_Manager.to(accelerator.device, dtype=weight_dtype)

    # for test: target img without augmentation
    test_img_input = Image.open(processed_image_path).convert('RGB') 
    test_aligned_img = test_img_input.crop((64, 70, 440, 446))  
    vit_face_recog_processor = ViTImageProcessor.from_pretrained(vit_face_recognition_model_name)
    test_vit_input = vit_face_recog_processor(images=test_aligned_img, return_tensors="pt")["pixel_values"][0] 
    test_vit_cls_output = vit_face_recognition_model(test_vit_input.unsqueeze(0).to(vit_face_recognition_model.device, dtype = torch.float32)).last_hidden_state[:, 0]
    img_name = os.path.basename(processed_image_path)[:-4]          



    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("identity_space")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    print("accelerator.num_processes", accelerator.num_processes)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")


    num_iter = 0
    trained_images_num = 0
    Embedding_Manager.train()
    for epoch in range(first_epoch, num_train_epochs):
        
        for step, batch in enumerate(train_dataloader):
            trained_images_num += total_batch_size
            

            with accelerator.accumulate(Embedding_Manager):
 
                vit_cls_output = vit_face_recognition_model(batch["vit_input"].to(dtype=weight_dtype)).last_hidden_state[:, 0]

                face_mask = batch["face_mask"].to(dtype=weight_dtype)
                hair_mask = batch["hair_mask"].to(dtype=weight_dtype)
                
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                
                # Get the text embedding for conditioning
                encoder_hidden_states = encode_prompt(batch["caption"], text_encoder, tokenizer, 
                                                        Embedding_Manager, is_train=True, 
                                                        face_img_embeddings = vit_cls_output, 
                                                        timesteps = timesteps)


                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents, sqrt_alpha_prod, sqrt_one_minus_alpha_prod = noise_scheduler.add_noise(latents, noise, timesteps)
                

                # Predict the noise residual
                model_pred = unet(noisy_latents.float(), timesteps, encoder_hidden_states.float()).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                
                
                
                # masked two-phase loss
                noise_indices = torch.where(timesteps > 600)[0]
                z0_indices = torch.where(timesteps <= 600)[0]
                
                # if timesteps <= 600
                if len(noise_indices) == 0:
                    fg_loss_noise = 0
                    hair_loss_noise = 0
                    noise_loss = 0
                else:
                    fg_loss_noise = F.mse_loss(face_mask[noise_indices] * model_pred[noise_indices].float(), face_mask[noise_indices] * target[noise_indices].float(), reduction="mean")
                    hair_loss_noise = F.mse_loss(hair_mask[noise_indices] * model_pred[noise_indices].float(), hair_mask[noise_indices] * target[noise_indices].float(), reduction="mean")
                    noise_loss = fg_loss_noise + hair_loss_noise * l_hair_diff_lambda

                # if timesteps > 600
                if len(z0_indices) == 0:
                    fg_loss_z0 = 0
                    hair_loss_z0 = 0
                    pred_z0_loss = 0
                else:
                    pred_z0 = (noisy_latents - sqrt_one_minus_alpha_prod * model_pred) / sqrt_alpha_prod                    
                    fg_loss_z0 = F.mse_loss(face_mask[z0_indices] * pred_z0[z0_indices] / vae.config.scaling_factor, \
                        face_mask[z0_indices] * latents[z0_indices] / vae.config.scaling_factor, reduction="mean")              
                    hair_loss_z0 = F.mse_loss(hair_mask[z0_indices] * pred_z0[z0_indices] / vae.config.scaling_factor, \
                        hair_mask[z0_indices] * latents[z0_indices] / vae.config.scaling_factor, reduction="mean")
                    pred_z0_loss = fg_loss_z0 + hair_loss_z0 * l_hair_diff_lambda
                    
                loss = noise_loss + pred_z0_loss
                
       
                
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(projection_params_to_optimize, max_grad_norm)
                optimizer_projection.step()
                lr_scheduler_proj.step()


                optimizer_projection.zero_grad(set_to_none=False)
                num_iter += 1

            if accelerator.sync_gradients:
                progress_bar.update(1)
                if global_step == max_train_steps-1:
                    print("saving embeddings...")
                    if accelerator.is_main_process:                    
                        try:
                            Embedding_Manager.save(test_vit_cls_output, pt_file_path, None)
                        except:
                            Embedding_Manager.module.save(test_vit_cls_output, pt_file_path, None)

                        logger.info(f"Saved state to {pt_file_path}")
                global_step += 1

            logs = {"imgs_num": trained_images_num, 
                    "loss": loss.detach().item(), 
                    "lr": lr_scheduler_proj.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)
            
            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()
