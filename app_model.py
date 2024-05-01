import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
import os
from utils import abs_path

# Base Class Definition
class BasePipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_scheduler(self):
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

    def model_train_img(self, img_path, output_dir, first_name, last_name):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_img(self, prompts, pt_path_1, pt_path_2=None):
        raise NotImplementedError("This method should be implemented by subclasses.")

# Subclass for Stable Diffusion 2.1
class StableDiffusion21(BasePipeline):
    def load_model(self):
        model_path = "stabilityai/stable-diffusion-2-1-base"
        return StableDiffusionPipeline.from_pretrained(model_path).to(self.device)

    def model_train_img(self, img_path, output_dir, first_name, last_name):
        from StableIdentity_model.train import train_img_to_embedding
        pt_path = os.path.join(output_dir, f"{first_name}_{last_name}.pt")
        train_img_to_embedding(img_path, abs_path(pt_path))
        return pt_path, None

    def generate_img(self, prompts, pt_paths_1, pt_path_2=None):
        self.pipe = self.load_model()
        self.set_scheduler()
        test_embeddings = [torch.load(pt_path).to(self.device) for pt_path in pt_paths_1]
        token_embeddings = [item for test_embedding in test_embeddings for item in (test_embedding[:, 0], test_embedding[:, 1])]
        tokens = [f"*v{i+1}" for i in range(len(token_embeddings))]
        self.pipe.tokenizer.add_tokens(tokens)
        token_ids = self.pipe.tokenizer.convert_tokens_to_ids(tokens)

        # resize token embeddings and set new embeddings
        self.pipe.text_encoder.resize_token_embeddings(len(self.pipe.tokenizer), pad_to_multiple_of = 8)
        for token_id, embedding in zip(token_ids, token_embeddings):
            self.pipe.text_encoder.get_input_embeddings().weight.data[token_id] = embedding
        
        images = []
        for prompt in prompts:
            images.append(self.pipe(prompt, guidance_scale = 8.5).images[0])
        return images

# Subclass for Stable Diffusion XL
class StableDiffusionXL(BasePipeline):
    def load_model(self):
        model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        return DiffusionPipeline.from_pretrained(model_path).to(self.device)

    def model_train_img(self, img_path, output_dir, first_name, last_name):
        from StableIdentity_model_SDXL.train import train_img_to_embedding
        pt_path_1 = os.path.join(output_dir, f"{first_name}_{last_name}_1.pt")
        pt_path_2 = os.path.join(output_dir, f"{first_name}_{last_name}_2.pt")
        train_img_to_embedding(img_path, abs_path(pt_path_1), abs_path(pt_path_2))

        return pt_path_1, pt_path_2

    def generate_img(self, prompts, pt_paths_1, pt_paths_2):
        self.pipe = self.load_model()
        self.set_scheduler()
        test_embeddings_1 = [torch.load(pt_path).to(self.device) for pt_path in pt_paths_1]
        test_embeddings_2 = [torch.load(pt_path).to(self.device) for pt_path in pt_paths_2]
        
        token_embeddings_1 = [item for test_embedding in test_embeddings_1 for item in (test_embedding[:, 0], test_embedding[:, 1])]
        token_embeddings_2 = [item for test_embedding in test_embeddings_2 for item in (test_embedding[:, 0], test_embedding[:, 1])]
        tokens = [f"*v{i+1}" for i in range(len(token_embeddings_1))]
        self.pipe.tokenizer.add_tokens(tokens)
        self.pipe.tokenizer_2.add_tokens(tokens)

        token_ids_1 = self.pipe.tokenizer.convert_tokens_to_ids(tokens)
        token_ids_2 = self.pipe.tokenizer_2.convert_tokens_to_ids(tokens)

        # resize token embeddings and set new embeddings
        self.pipe.text_encoder.resize_token_embeddings(len(self.pipe.tokenizer), pad_to_multiple_of = 8)
        for token_id, embedding in zip(token_ids_1, token_embeddings_1):
            self.pipe.text_encoder.get_input_embeddings().weight.data[token_id] = embedding

        self.pipe.text_encoder_2.resize_token_embeddings(len(self.pipe.tokenizer_2), pad_to_multiple_of = 8)
        for token_id, embedding in zip(token_ids_2, token_embeddings_2):
            self.pipe.text_encoder_2.get_input_embeddings().weight.data[token_id] = embedding
        
        images = []
        for prompt in prompts:
            images.append(self.pipe(prompt, guidance_scale = 8.5).images[0])
        return images

# Factory function
def get_model(model_name):
    if model_name == "stable-diffusion-2-1":
        return StableDiffusion21()
    elif model_name == "stable-diffusion-xl":
        return StableDiffusionXL()
    else:
        raise ValueError(f"Model type {model_name} not supported")