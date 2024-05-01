# Identity_Preserved_Images_Generation

The project is the Columbia University Applied Computer Vision Course project. We developed an application that can automatically transform personal stories into customized comic strips preserving the faces of users and their friends in just minutes! The method is inspired and extended by the paper [StableIdentity: Inserting Anybody into Anywhere at First Sight](https://arxiv.org/abs/2401.15975)

## Getting Started

### Installation

1. **Create and Activate the Environment**:
   - To create a new Conda environment named `myenv` and install Python, open your terminal and run:
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

### Running the Demo with GUI

To run the demo, execute the following command in the root directory of the repository in your terminal:

```bash
python3 app.py
```

A link will appear in the terminal output. Click on this link to access the application in your web browser.

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

Replace `"SDXL"` with `"SD"` or `"failed"` depending on which model you want to use. The script will automatically import and use the appropriate training function based on your model choice.

This setup ensures that you can easily switch between different models for training depending on the project requirements or testing scenarios.