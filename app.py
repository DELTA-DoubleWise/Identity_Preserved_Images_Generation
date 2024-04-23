import gradio as gr
import torch
from prompt_processing import story_to_prompts, prompts_parse
import re
import os
from app_model import get_model
from util import abs_path

class GradioApp:
    def __init__(self):
        self.image_output_dir = "runtime/img"
        self.embedding_output_dir = "runtime/face_embeddings"
        self.display_images = []
        self.pt_paths_1 = []
        self.pt_paths_2 = []
        self.name_list = []
        self.meta_data = {}
        self.model_name = "stable-diffusion-xl"   # Could also be "stable-diffusion-2-1" "stable-diffusion-xl"
        self.model = get_model(self.model_name)
        self.pretrained_images_dict = {
            'Taylor Swift': ('pretrained_images/Taylor_Swift.png', 'pretrained_images/Taylor_Swift.pt'),
            'Jane Smith': ('pretrained_images/Jane_Smith.png', 'pretrained_images/Jane_Smith.pt')
        }
        self.pretrained_images = [
            ["pretrained_images/Taylor_Swift.png", "Taylor Swift"],
            ["pretrained_images/Jane_Smith.png", "Jane Smith"]
        ]
        self.style_options = ["2d minimalistic", "4k", "8k", "cartoon", "chiaroscuro lighting technique", "cinematic",
                              "clipart style", "close-up", "comic style", "cyberpunk", "detailed", "digitally enhanced",
                              "disney style", "dramatic", "dreamy", "expressive", "fantasy art", "high contrast",
                              "high resolution", "highly detailed", "intimate", "low-res", "medium shot", "minimalistic",
                              "oil painting", "photorealistic", "pixel-art", "psychedelic style", "smooth light",
                              "studio ghibli style", "surreal", "Van Gogh", "vaporwave style", "vector", "vivid colors",
                              "watercolor painting", "wide shot"]

    def process_images(self, image_files, names):
        os.makedirs(self.image_output_dir, exist_ok=True)
        os.makedirs(self.embedding_output_dir, exist_ok=True)
        for image_tuple, name in zip(image_files, names.split(",")):  # Assuming names are comma-separated
            first_name, last_name = name.strip().split(" ")
            image = image_tuple[0]
            processed_image_path = os.path.join(self.image_output_dir, f"{first_name}_{last_name}.png")
            image.save(processed_image_path, 'PNG')
            print(abs_path(processed_image_path), self.embedding_output_dir, first_name, last_name)
            pt_file_path_1, pt_file_path_2 = self.model.model_train_img(abs_path(processed_image_path), self.embedding_output_dir, first_name, last_name)
            # pt_file_path_1 = os.path.join(self.embedding_output_dir,f"{first_name}_{last_name}_1.pt")
            # pt_file_path_2 = os.path.join(self.embedding_output_dir,f"{first_name}_{last_name}_2.pt")
            # pt_file_path_1 = os.path.join(self.embedding_output_dir,f"{first_name}_{last_name}.pt")
            # pt_file_path_2 = None
            self.display_images.append(processed_image_path)
            self.pt_paths_1.append(pt_file_path_1)
            self.pt_paths_2.append(pt_file_path_2)
            self.name_list.append(name.strip())

        return [(img, name) for img, name in zip(self.display_images, self.name_list)], ", ".join(self.name_list)

    def process_text(self, input_text):
        self.meta_data = {self.name_list[i]: {"signal_word": f"*v{2*i+1} *v{2*i+2}"} for i in range(len(self.name_list))}
        processed_text = story_to_prompts(input_text, self.meta_data)
        return processed_text

    def load_model_and_generate_images(self, processed_text, style_dropdown):
        processed_text_list = [prompt for prompt in re.findall(r'"(.*?)"', processed_text)]
        prompts = prompts_parse(processed_text, self.meta_data, style_dropdown)
        
        images = self.model.generate_img(prompts, self.pt_paths_1, self.pt_paths_2)

        return [(images[i], processed_text_list[i]) for i in range(len(processed_text_list))]

    def add_pretrained_image(self, clicked_event):
        name = clicked_event[1]
        pretrained_image_path, pretrained_pt_path = self.pretrained_images_dict[name]
        self.display_images.append(pretrained_image_path)
        self.pt_paths_1.append(pretrained_pt_path)  # Assuming .pt paths handling similar to images
        self.name_list.append(name.strip())
        return [(img, name) for img, name in zip(self.display_images, self.name_list)], ", ".join(self.name_list)

    def launch(self):
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    image_input = gr.Gallery(label="Upload Images", type="image", show_label=False)
                    name_input = gr.Textbox(label="Enter Names")
                    upload_btn = gr.Button("Upload")

                    # Pre-trained images dataset and interaction
                    pretrained_dataset = gr.Dataset(components=[gr.Image(type="filepath", visible=False), gr.Text(visible=False)],
                                                    samples=self.pretrained_images,
                                                    label="Pre-trained Images (Click to Add)")

                with gr.Column():
                    processed_image_display = gr.Gallery(label="Uploaded Images with Trained Face Embeddings")
                    name_list_display = gr.Label(label="Names of Uploaded Images")
                    text_input = gr.Textbox(label="Enter Story Prompt Using the Names Uploaded")
                    processed_text_output = gr.Textbox(label="Processed Story, Please Revise Based on Your Preference")
                    text_process_btn = gr.Button("Process Text")

                    style_dropdown = gr.Dropdown(choices=self.style_options, label="Add styles to prompts", multiselect=True)
                    final_images_display = gr.Gallery(label="Final Images")
                    generate_images_btn = gr.Button("Generate Images")

                upload_btn.click(self.process_images, inputs=[image_input, name_input], outputs=[processed_image_display, name_list_display])
                text_process_btn.click(self.process_text, inputs=[text_input], outputs=processed_text_output)
                generate_images_btn.click(self.load_model_and_generate_images, inputs=[processed_text_output, style_dropdown], outputs=final_images_display)
                pretrained_dataset.click(self.add_pretrained_image, inputs=[pretrained_dataset],
                                             outputs=[processed_image_display, name_list_display])

            demo.launch(share=True)

if __name__ == "__main__":
    app = GradioApp()
    app.launch()
