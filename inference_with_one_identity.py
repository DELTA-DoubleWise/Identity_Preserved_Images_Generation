import argparse
import torch
import os
from transformers import ViTModel, ViTImageProcessor
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
from tqdm import tqdm
from PIL import Image

# Set device
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def load_model(model_type):
    if model_type == "SD2.1":
        model_path = "stabilityai/stable-diffusion-2-1-base"
    elif model_type == "SDXL":
        model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        raise ValueError("Unsupported model type provided. Choose either 'SD2.1' or 'SDXL'.")

    # Load diffusion model
    pipe = DiffusionPipeline.from_pretrained(model_path)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe

def run_inference(pipe, prompt, model_name, emb_path_1, emb_path_2=None, output = "./output.png"):
    tokens = ["v1*", "v2*"]

    embeddings = torch.load(emb_path_1).cuda()
    v_embeddings = [embeddings[:, i] for i in range(2)]

    # Add tokens and set new embeddings
    pipe.tokenizer.add_tokens(tokens)
    token_ids = pipe.tokenizer.convert_tokens_to_ids(tokens)

    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer), pad_to_multiple_of=8)
    for token_id, embedding in zip(token_ids, v_embeddings):
        pipe.text_encoder.get_input_embeddings().weight.data[token_id] = embedding

    if(model_name == "SDXL"):
        embeddings_2 = torch.load(emb_path_2).cuda()
        v_embeddings_2 = [embeddings_2[:, i] for i in range(2)]

        pipe.tokenizer_2.add_tokens(tokens)
        token_ids_2 = pipe.tokenizer_2.convert_tokens_to_ids(tokens)
        pipe.text_encoder_2.resize_token_embeddings(len(pipe.tokenizer_2), pad_to_multiple_of = 8)
        for token_id, embedding in zip(token_ids_2, v_embeddings_2):
            pipe.text_encoder_2.get_input_embeddings().weight.data[token_id] = embedding
    

    image = pipe(prompt, guidance_scale=8.5).images[0]
    image.save(output)
    print(f"Inference completed and image saved as {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Stable Diffusion models.")
    parser.add_argument("--model_type", type=str, required=True, choices=["SD2.1", "SDXL"], help="Model type: 'SD2.1' or 'SDXL'")
    parser.add_argument("--emb_path_1", type=str, required=True, help="Path to the first embedding file")
    parser.add_argument("--emb_path_2", type=str, default=None, help="Path to the second embedding file (optional)")
    parser.add_argument("--output_path", type=str, default="./output.png", help="Path to the output file")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generating the image")
    args = parser.parse_args()

    pipe = load_model(args.model_type)
    run_inference(pipe, args.prompt, args.model_type, args.emb_path_1, args.emb_path_2, args.output_path)
