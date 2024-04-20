import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from dataset import FaceDataset
from model import IDPreservedGenerativeModel
from torch.optim import Adam
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from util import add_noise_return_paras
import types
import os

DTYPE = torch.float16
MAX_TRAINING_STEP = 5001

def train_model(model, data_loader, preprocessed_image, save_path, device, num_epochs=5, learning_rate=5e-5):
    """
    Trains the IDPreservedGenerativeModel model.

    Parameters:
    - model: The IDPreservedGenerativeModel instance to be trained.
    - data_loader: DataLoader providing the dataset for training.
    - device: The device (CPU or CUDA) to train the model on.
    - num_epochs: Number of epochs to train the model for.
    - learning_rate: Learning rate for the optimizer.
    """
    model = model.to(device)

    learning_rate = 5e-5
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08
    lr_scheduler = "constant"
    gradient_accumulation_steps = 1
    max_train_steps = MAX_TRAINING_STEP
    lr_num_cycles = 1
    lr_power = 1.0
    
    optimizer_projection = torch.optim.AdamW(
        model.face_projection_layer.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    lr_scheduler_proj = get_scheduler(
        lr_scheduler,
        optimizer=optimizer_projection,
        num_warmup_steps=0,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )
    # optimizer = Adam(model.face_projection_layer.parameters(), lr=learning_rate)
    scaler = GradScaler()
    global_step = 0
    print("Start Training...")
    
    # Training loop
    for epoch in range(num_epochs):
        model.face_projection_layer.train()  # Set the model to training mode
        total_loss = 0

        for batch in tqdm(data_loader):
            if global_step>MAX_TRAINING_STEP:
                break
                
            global_step += 1
            image_embedding = batch["vit_output"].to(device, dtype = DTYPE)
            face_mask = batch["mask_face"].to(device, dtype = DTYPE)
            hair_mask = batch["mask_hair"].to(device, dtype = DTYPE)
            pixel_values = batch["face_img"].to(device, dtype = DTYPE)
            prompt = batch['text']
    
            optimizer_projection.zero_grad()
            # Forward pass with mixed precision
            with autocast():
                model_pred, loss = model(
                    prompt=prompt,
                    pixel_values=pixel_values,
                    image_embedding=image_embedding,
                    face_mask=face_mask,
                    hair_mask=hair_mask,
                    device=device
                )
            
            # print(loss.item())
            # Backward pass and optimize
            scaler.scale(loss).backward()  # Scale the loss before the backward pass
            scaler.step(optimizer_projection)  # Adjust the optimizer's step
            scaler.update()  # Update the scale for next iteration
            lr_scheduler_proj.step()  # Step the learning rate scheduler

            total_loss += loss.item()

            if global_step == max_train_steps:
                avg_loss = total_loss / (max_train_steps-epoch*len(data_loader))
                model.save(save_path)
        model.save(f"test/00001_{epoch}.pt")
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        
def train_image(img_path, pt_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    print("Creating Dataset...")
    dataset = FaceDataset(img_path, device)
    original_img_vit_cls_output = dataset.get_vit_cls_output()
    print("Dataset Created!")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Load and prepare your model
    print("Loading Pretrained Model...")
    model = IDPreservedGenerativeModel.from_pretrained("stabilityai/stable-diffusion-2-1",torch_dtype=torch.float32)
    model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
    model.scheduler.add_noise = types.MethodType(add_noise_return_paras, model.scheduler)
    model.load_adaptor(device=device, face_image_embedding = original_img_vit_cls_output)
    # model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
    
    # Call the training function
    train_model(model, data_loader, original_img_vit_cls_output, pt_file_path, device)
    
    
if __name__ == "__main__":
    ## test image
    face_img_path = 'test/00001.png'  # replace it with real path 
    pt_file_path = 'test/00001.pt'
    train_image(face_img_path, pt_file_path)
    


