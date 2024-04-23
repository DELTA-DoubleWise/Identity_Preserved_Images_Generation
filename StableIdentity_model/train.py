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
from StableIdentity_model.models.celeb_embeddings import embedding_forward  
from StableIdentity_model.models.embedding_manager import EmbeddingManagerId_adain
from StableIdentity_model.datasets_face.face_id import FaceIdDataset
from StableIdentity_model.utils import text_encoder_forward, set_requires_grad, latents_to_images, latents_to_images_tensor, add_noise_return_paras
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
    
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:  
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train_img_to_embedding(processed_image_path, pt_file_path):

    print(f"image path: {processed_image_path}")
    print(f"embedding saving path: {pt_file_path}")
    pretrained_model_name = "stabilityai/stable-diffusion-2-1-base"
    vit_face_recognition_model_name = "jayanta/google-vit-base-patch16-224-face"
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
    max_train_steps = 501
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

        total_loss = 0
        
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
                    # print(vae.config.scaling_factor)
                    # print(pred_z0.max(), pred_z0.min(), pred_z0.mean(), pred_z0.std())
                    # print(latents.max(), latents.min(), latents.mean(), latents.std())
                    
                loss = noise_loss + pred_z0_loss

                print(f"noise loss: {noise_loss}, pred z0 loss: {pred_z0_loss}")
                # print(f"hair_loss_noise: {fg_loss_noise}, hair_loss_noise: {hair_loss_noise}")
                # print(f"fg loss z0: {fg_loss_z0}, hair_loss_z0: {hair_loss_z0}")
                
                # print(pred_z0.max(), pred_z0.min(), pred_z0.mean(), pred_z0.std())
                # print(latents.max(), latents.min(), latents.mean(), latents.std())
    
                
                total_loss += loss.item()
                
                
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
                if global_step % 100 == 0 and global_step!=0:
                    avg_loss = total_loss / (global_step - epoch*len(train_dataloader))
                    print(f"Average Loss: {avg_loss:.4f}")
                global_step += 1

            logs = {"imgs_num": trained_images_num, 
                    "loss": loss.detach().item(), 
                    "lr": lr_scheduler_proj.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)
            
            if global_step >= max_train_steps:
                break
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}], Average Loss: {avg_loss:.4f}")

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()