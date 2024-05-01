# Identity_Preserved_Images_Generation

The project is the Columbia University Applied Computer Vision Course project. We developed an application that can automatically transform personal stories into customized comic strips preserving the faces of users and their friends in just minutes! The method is inspired and extended by the paper [StableIdentity: Inserting Anybody into Anywhere at First Sight](https://arxiv.org/abs/2401.15975)

## Getting Started

### Installation

1. **Create and Activate the Environment**:
   - To create a new Conda environment named `IPCSG` and install Python, open your terminal and run:
     ```bash
     conda create --name IPCSG
     ```
   - Activate the new environment:
     ```bash
     conda activate IPCSG
     ```

2. **Install Project Dependencies**:
   - Ensure all dependencies specific to our project are installed using the `requirements.txt` file:
     ```bash
     conda install --file requirements.txt
     ```
   - If any packages are missing in the Conda repositories, use pip to install them:
     ```bash
     pip install -r requirements.txt
     ```

3. **Download Pretrained model**:

    - Download the [face parsing model](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view?pli=1) into StableIdentity_model/models/face_parsing/res/cp and StableIdentity_model_SDXL/models/face_parsing/res/cp.

---

### Running the Demo with GUI

To run the demo, execute the following command in the root directory of the repository in your terminal:

```bash
python3 app.py
```

A link will appear in the terminal output. Click on this link to access the application in your web browser.

---

### Running the Training Script Without GUI

This script allows you to process an image and generate embeddings using one of three different models. Specify the image file, output paths for the embeddings, and the model you wish to use.

**Usage:**

```bash
python train_embeddings.py --image_path "path/to/your/image.png" --pt_path_1 "path/to/output/embedding_1.pt" --pt_path_2 "path/to/output/embedding_2.pt" --model SDXL
```

**Example:**

```bash
python train_embeddings.py --image_path "test/00001.png" --pt_path_1 "test/00001_SDXL_1.pt" --pt_path_2 "test/00001_SDXL_2.pt" --model SDXL
```

Replace `"SDXL"` with `"SD21"` or `"failed"` depending on which model you want to use. The script will automatically import and use the appropriate training function based on your model choice.

This setup ensures that you can easily switch between different models for training depending on the project requirements or testing scenarios.


---

### Running the Inference Script Without GUI

The `inference_with_one_identity.py` script allows you to generate images using the Stable Diffusion 2.1 or Stable Diffusion XL models with pre-trained embeddings without using GUI. 


**Usage:**

Run the script from the command line by specifying the model type, the paths to embedding files, and your image generation prompt. The second embedding path is optional and only needed for the SDXL model.

```bash
python inference_with_one_identity.py --model_type [MODEL_TYPE] --emb_path_1 [PATH_TO_EMB1] --emb_path_2 [PATH_TO_EMB2] --prompt [PROMPT]
```

**Parameters:**
- `--model_type`: Specifies the model to use. Accepts `"SD2.1"` for Stable Diffusion 2.1 or `"SDXL"` for Stable Diffusion XL.
- `--emb_path_1`: Path to the first embedding file. Required.
- `--emb_path_2`: Path to the second embedding file. Optional, only needed for SDXL.
- `--prompt`: The text prompt to guide the image generation.

**Example:**
To generate an image with the SDXL model using a custom prompt:

```bash
python inference_with_one_identity.py --model_type SDXL --emb_path_1 path/to/first_embedding.pt --emb_path_2 path/to/second_embedding.pt --prompt "A futuristic cityscape, vibrant and detailed"
```

To use the SD2.1 model, you can omit the `--emb_path_2` parameter:

```bash
python inference_with_one_identity.py --model_type SD2.1 --emb_path_1 path/to/first_embedding.pt --prompt "A serene landscape with a sunset background"
```
---
