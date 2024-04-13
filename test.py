import torch
from diffusers import DPMSolverMultistepScheduler
from model import IDPreservedGenerativeModel

DTYPE = torch.float16
def main():

    # Initialize the Diffusion Model
    model = IDPreservedGenerativeModel.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Example input data
    prompt = "A photo of an v1* v2* riding a horse"
    pixel_values = torch.randn(1, 3, 256, 256)  # Dummy image tensor
    image_embedding = torch.randn(1, 768)  # Dummy image embedding
    face_mask = torch.ones(1, 32, 32, dtype=torch.long)  # Dummy mask
    hair_mask = torch.ones(1, 32, 32, dtype=torch.long)  # Dummy mask

    pixel_values = pixel_values.to(device, dtype=DTYPE)
    image_embedding = image_embedding.to(device, dtype=DTYPE)
    face_mask = face_mask.to(device, dtype=DTYPE)
    hair_mask = hair_mask.to(device, dtype=DTYPE)

    model = model.to(device)
    model.load_adaptor(device=device)

    # Run the model (forward pass)
    model_pred, loss = model(
        prompt=prompt,
        pixel_values=pixel_values,
        image_embedding=image_embedding,
        face_mask=face_mask,
        hair_mask=hair_mask,
        device=device
    )

    print(f"Model prediction shape: {model_pred.shape}")
    print(f"Loss: {loss}")


if __name__ == "__main__":
    main()
