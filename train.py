import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import DPMSolverMultistepScheduler
from dataset import FaceDataset
from model import IDPreservedGenerativeModel
from torch.optim import Adam
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import os

DTYPE = torch.float16
MAX_TRAINING_STEP = 501

def train_model(model, data_loader, dataset, device, num_epochs=1, learning_rate=5e-5):
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
    optimizer = Adam(model.face_projection_layer.parameters(), lr=learning_rate)
    scaler = GradScaler()
    global_step = 0
    print("Start Training...")
    original_img_vit_cls_output = dataset.get_vit_cls_output()
    # Training loop
    for epoch in range(num_epochs):
        model.face_projection_layer.train()  # Set the model to training mode
        total_loss = 0

        for batch in tqdm(data_loader):
            if global_step>=MAX_TRAINING_STEP:
                break
                
            global_step += 1
            image_embedding = batch["vit_output"].to(device, dtype = DTYPE)
            face_mask = batch["mask_face"].to(device, dtype = DTYPE)
            hair_mask = batch["mask_hair"].to(device, dtype = DTYPE)
            pixel_values = batch["face_img"].to(device, dtype = DTYPE)
            prompt = batch['text']
    
            optimizer.zero_grad()
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
            scaler.step(optimizer)  # Adjust the optimizer's step
            scaler.update()  # Update the scale for next iteration

            total_loss += loss.item()
            
            save_path = os.path.join("./test_result",f"{dataset.get_img_name()}-{global_step}.pt")
            if global_step%100==0:
                model.save(original_img_vit_cls_output, save_path)

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

## test image
face_img_path = '00001.png'  # replace it with real path 
    
print("Creating Dataset...")
dataset = FaceDataset(face_img_path, device)
print("Dataset Created!")
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load and prepare your model
print("Loading Pretrained Model...")
model = IDPreservedGenerativeModel.from_pretrained("stabilityai/stable-diffusion-2-1",torch_dtype=torch.float32)
model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
model.load_adaptor(device=device)

# Call the training function
train_model(model, data_loader, dataset, device, num_epochs=1, learning_rate=5e-5)
