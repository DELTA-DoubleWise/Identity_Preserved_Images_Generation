import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import DPMSolverMultistepScheduler
from dataset import FaceDataset
from model import IDPreservedGenerativeModel
from torch.optim import Adam
from tqdm import tqdm


def train_model(model, data_loader, device, num_epochs=10, learning_rate=1e-4):
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
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for batch in tqdm(data_loader):
            pixel_values, prompt, image_embedding, face_mask, hair_mask = [item.to(device) for item in batch]

            # Forward pass
            model_pred, loss = model(
                prompt=prompt,
                pixel_values=pixel_values,
                image_embedding=image_embedding,
                face_mask=face_mask,
                hair_mask=hair_mask,
                device=device
            )

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Optional: Save the model after each epoch
        model.save_pretrained(f"model_epoch_{epoch+1}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

face_img_path = './data'  # replace it with real path
dataset = FaceDataset(face_img_path, device)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load and prepare your model
model = IDPreservedGenerativeModel.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32)
model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
model.load_adaptor(device=device)

# Call the training function
train_model(model, data_loader, device, num_epochs=10, learning_rate=1e-4)
