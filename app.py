import gradio as gr
import torch
from prompt_processing import story_to_prompts, prompts_parse
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
import re

pt_paths = []
name_list = []
meta_data = {}
model_path = "stabilityai/stable-diffusion-2-1-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained(model_path)
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)


def process_images(image_files, names):
    display_images = []
    global pt_paths
    global name_list
    pt_paths = []
    name_list = []
    # for image, name in zip(image_files, names.split(",")):  # Assuming names are comma-separated
    #     # Here you would process your image and generate a .pt file
    #     first_name, last_name = name.strip().split(" ")  # Split first and last name
    #     processed_image_path = f"{"img/"+first_name+"_"+last_name}.png"  # Placeholder for image saving path
    #     pt_file_path = f"{"face_embeddings"+first_name+"_"+last_name}.pt"  # Placeholder for .pt file path
    #     processed_images.append(processed_image_path)  # Store processed image path
    #     name_list.append(name.strip())  # Store names
    display_images = ["img/Xuezhen_Wang.jpg", "img/Taylor_Swift.jpg", "img/Yucheng_Wang.jpg"]
    name_list = ["Xuezhen Wang", "Taylor Swift", "Yucheng Wang"]
    pt_paths = ["face_embeddings/Xuezhen_Wang.pt", "face_embeddings/Taylor_Swift.pt", "face_embeddings/Yucheng_Wang.pt"]
    return [(img, name) for img, name in zip(display_images, name_list)], ", ".join(name_list)

def process_text(input_text):
    global meta_data
    meta_data = {name_list[i]: {"signal_word": f"*v{2*i+1} *v{2*i+2}"} for i in range(len(name_list))}
    processed_text = story_to_prompts(input_text, meta_data)
    return processed_text

def load_model_and_generate_images(processed_text):
    processed_text_list = [prompt for prompt in re.findall(r'"(.*?)"', processed_text)]
    global pt_paths
    prompts = prompts_parse(processed_text, meta_data)
    test_embeddings = [torch.load(pt_path).to(device) for pt_path in pt_paths]
    token_embeddings = [item for test_embedding in test_embeddings for item in (test_embedding[:, 0], test_embedding[:, 1])]
    tokens = [f"*v{i+1}" for i in range(len(token_embeddings))]
    pipe.tokenizer.add_tokens(tokens)
    token_ids = pipe.tokenizer.convert_tokens_to_ids(tokens)

    # resize token embeddings and set new embeddings
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer), pad_to_multiple_of = 8)
    for token_id, embedding in zip(token_ids, token_embeddings):
        pipe.text_encoder.get_input_embeddings().weight.data[token_id] = embedding


    images = []
    for prompt in prompts:
        images.append(pipe(prompt, guidance_scale = 8.5).images[0])
    return [(images[i], processed_text_list[i]) for i in range(len(processed_text_list))]
    return images

def gradio_app():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image_input = gr.Gallery(label="Upload Images", type="image", show_label=False)
                name_input = gr.Textbox(label="Enter Names (name must be of the format 'FirstName LastName') in the order of the uploaded images", placeholder="John Doe, Jane Smith")
                upload_btn = gr.Button("Upload")
            with gr.Column():
                processed_image_display = gr.Gallery(label="Uploaded Images with Trained Face Embeddings")
                name_list_display = gr.Label(label="Names of Uploaded Images")

        text_input = gr.Textbox(label="Enter Stroy Prompt Using the Names Uploaded", placeholder="John Doe and Jane Smith went on a picnic.")
        processed_text_output = gr.Textbox(label="LLM Processed Story, Please Revise Based on Your Preference", interactive=True)
        text_process_btn = gr.Button("Process Text")

        final_images_display = gr.Gallery(label="Final Images")
        generate_images_btn = gr.Button("Generate Comics")

        # Define interactions
        def update_gallery_and_names(processed_image_names, name_list):
            processed_image_display.update(processed_image_names)
            name_list_display.update("\n".join(name_list))

        upload_btn.click(process_images, inputs=[image_input, name_input], outputs=[processed_image_display, name_list_display])
        text_process_btn.click(process_text, inputs=[text_input], outputs=processed_text_output)
        generate_images_btn.click(load_model_and_generate_images, inputs=[processed_text_output], outputs=final_images_display)

    demo.launch(share=True)


if __name__ == "__main__":
    gradio_app()