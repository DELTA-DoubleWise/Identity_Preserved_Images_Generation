import gradio as gr
import torch
from prompt_processing import story_to_prompts, prompts_parse
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
from StableIdentity_model.train import train_img_to_embedding
import re
import os

selected_keywords = []
pt_paths = []
name_list = []
meta_data = {}
model_path = "stabilityai/stable-diffusion-2-1-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained(model_path)
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

def abs_path(rel_path):
    dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the script
    abs_file_path = os.path.join(dir_path, rel_path)
    return abs_file_path


def process_images(image_files, names):
    display_images = []
    global pt_paths
    global name_list
    pt_paths = []
    name_list = []
    for image_tuple, name in zip(image_files, names.split(",")):  # Assuming names are comma-separated
        # Here you would process your image and generate a .pt file
        first_name, last_name = name.strip().split(" ")  # Split first and last name
        image = image_tuple[0]
        os.makedirs("runtime/img", exist_ok=True)
        processed_image_path = os.path.join("runtime/img",f"{first_name}_{last_name}.png")  # Placeholder for image saving path
        image.save(processed_image_path, 'PNG')
        os.makedirs("runtime/face_embeddings", exist_ok=True)
        pt_file_path = os.path.join("runtime/face_embeddings",f"{first_name}_{last_name}.pt")  # Placeholder for .pt file path

        train_img_to_embedding(abs_path(processed_image_path), abs_path(pt_file_path))
        display_images.append(processed_image_path)  # Store processed image path
        pt_paths.append(pt_file_path)
        name_list.append(name.strip())  # Store names
    # display_images = ["img/Xuezhen_Wang.jpg", "img/Taylor_Swift.jpg", "img/Yucheng_Wang.jpg"]
    # name_list = ["Xuezhen Wang", "Taylor Swift", "Yucheng Wang"]
    # pt_paths = ["face_embeddings/Xuezhen_Wang.pt", "face_embeddings/Taylor_Swift.pt", "face_embeddings/Yucheng_Wang.pt"]
    return [(img, name) for img, name in zip(display_images, name_list)], ", ".join(name_list)

def process_text(input_text):
    global meta_data
    meta_data = {name_list[i]: {"signal_word": f"*v{2*i+1} *v{2*i+2}"} for i in range(len(name_list))}
    processed_text = story_to_prompts(input_text, meta_data)
    return processed_text

def load_model_and_generate_images(processed_text, style_dropdown):
    processed_text_list = [prompt for prompt in re.findall(r'"(.*?)"', processed_text)]
    global pt_paths
    prompts = prompts_parse(processed_text, meta_data, style_dropdown)
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
                
                # A gr.Dataset component for pre-trained images
                global pretrained_images_dict
                pretrained_images_dict = {'Taylor Swift':('pretrained_images/Taylor_Swift.png','pretrained_images/Taylor_Swift.pt'),
                    'Jane Smith': ('pretrained_images/Jane_Smith.png','pretrained_images/Jane_Smith.pt')
                     # Add more pre-trained images and names
                    }
                pretrained_images = [["pretrained_images/Taylor_Swift.png","Taylor Swift"], 
                    ["pretrained_images/Jane_Smith.png", "Jane Smith"]
                     # Add more pre-trained images and names
                ]    
                pretrained_dataset = gr.Dataset(components=[gr.Image(type="filepath"), gr.Text()], 
                                samples=pretrained_images,
                                label="Pre-trained Images (Click to Add)")
                
                # pretrained_images = [("pretrained_images/image1.jpg", "John Doe"),
                #                      ("pretrained_images/image2.jpg", "Jane Smith"),
                #                      # Add more pre-trained images and names
                #                     ]
                # pretrained_dataset = gr.Dataset(components=[gr.Image(), gr.Text()], 
                #                                 samples=pretrained_images,
                #                                 label="Pre-trained Images (Click to Add)")
                
                
            with gr.Column():
                processed_image_display = gr.Gallery(label="Uploaded Images with Trained Face Embeddings")
                name_list_display = gr.Label(label="Names of Uploaded Images")
                text_input = gr.Textbox(label="Enter Story Prompt Using the Names Uploaded", placeholder="John Doe and Jane Smith went on a picnic.")
                processed_text_output = gr.Textbox(label="LLM Processed Story, Please Revise Based on Your Preference", interactive=True)
                text_process_btn = gr.Button("Process Text")
                
                # A dropdown menu
                style_options = ["comic style", "4k", "Van Gogh", "oil painting", "vivid colors"]
                style_dropdown = gr.Dropdown(choices=style_options, label="Select a style")
                
                final_images_display = gr.Gallery(label="Final Images")
                generate_images_btn = gr.Button("Generate Comics")
                
        def add_pretrained_image(clicked_event):
            # print(image_path,type(image_path))
            name = clicked_event[1]

            pretrained_image_path, pretrained_pt_path = pretrained_images_dict[name]
            # os.makedirs("runtime/face_embeddings", exist_ok=True)
            # pt_file_path = f"runtime/face_embeddings/{name.replace(' ', '_')}.pt"
            # train_img_to_embedding(abs_path(processed_image_path), abs_path(pt_file_path))
            name_list.append(name.strip())  # Store names
            pt_paths.append(pretrained_pt_path)
            # processed_image_display.value = 
            # name_list_display.value = ", ".join(name_list)
            
            return (processed_image_display.value + [(pretrained_image_path, name)]), ", ".join(name_list)

            # processed_image_display.value = .append(pretrained_image_path)  # Store processed image path
            # pt_paths.append(pretrained_pt_path)
            
            # name_list.append(name)
            # processed_image_display.update((pretrained_image_path, name))
            # name_list_display.update(", ".join(name_list))

        # Define interactions
        def update_gallery_and_names(processed_image_names, name_list):
            processed_image_display.update(processed_image_names)
            name_list_display.update("\n".join(name_list))
        
        pretrained_dataset.click(add_pretrained_image, inputs=[pretrained_dataset],
                                 outputs=[processed_image_display, name_list_display])
        upload_btn.click(process_images, inputs=[image_input, name_input], outputs=[processed_image_display, name_list_display])
        text_process_btn.click(process_text, inputs=[text_input], outputs=processed_text_output)
        generate_images_btn.click(load_model_and_generate_images, inputs=[processed_text_output, style_dropdown], outputs=final_images_display)
        
    demo.launch(share=True)


if __name__ == "__main__":
    gradio_app()
