import argparse
from pathlib import Path

def get_training_function(model_choice):
    """
    Dynamically imports the training function based on the model choice.
    
    Args:
        model_choice (str): The model selection from the command line.
    
    Returns:
        function: The train_img_to_embedding function from the selected model module.
    """
    if model_choice == 'SDXL':
        from StableIdentity_model_SDXL.train import train_img_to_embedding
    elif model_choice == 'SD21':
        from StableIdentity_model.train import train_img_to_embedding
    elif model_choice == 'failed':
        from failed_model.train import train_img_to_embedding
    else:
        raise ValueError("Invalid model choice. Please choose 'SDXL', 'SD', or 'failed'.")
    
    return train_img_to_embedding

def main(image_path, pt_path_1, pt_path_2, model_choice):
    train_img_to_embedding = get_training_function(model_choice)
    
    if not Path(image_path).exists():
        raise FileNotFoundError(f"The specified image file was not found: {image_path}")
    
    # Run the training function
    if pt_path_2 is not None:
        train_img_to_embedding(image_path, pt_path_1, pt_path_2)
        print(f"Embeddings generated and saved at {pt_path_1} and {pt_path_2}")
    else: 
        train_img_to_embedding(image_path, pt_path_1)
        print(f"Embeddings generated and saved at {pt_path_1}")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image and generate embeddings without using the GUI.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--pt_path_1", type=str, required=True, help="Path to save the first embedding output.")
    parser.add_argument("--pt_path_2", type=str, default=None, help="Path to save the second embedding output (optional)")
    parser.add_argument("--model", type=str, required=True, choices=['SDXL', 'SD21', 'failed'], help="Model choice: 'SDXL', 'SD21', or 'failed'.")
    args = parser.parse_args()

    main(args.image_path, args.pt_path_1, args.pt_path_2, args.model)
