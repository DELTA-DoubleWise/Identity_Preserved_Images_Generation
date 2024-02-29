import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_pil_image, to_tensor
import random


def get_mask_from_parsing(parsing, parse_model_name="jonathandinu/face-parsing"):
    """
    Generate masks for the whole face (including hair, hat, and earring) and hair from the parsing output.

    Parameters:
    - parsing: A tensor with shape (H, W) containing the parsing labels.

    Returns:
    - face_mask: A boolean mask for the whole face including hair, hat, and earring.
    - hair_mask: A boolean mask for the hair.
    """
    if parse_model_name == "jonathandinu/face-parsing":
        # Labels from 1 to 15 for the face, including skin, facial features, hair, hat, and earring
        face_labels = list(range(1, 16))

        # Define label for hair
        hair_label = 13

        # Generate face mask
        face_mask = torch.zeros_like(parsing, dtype=torch.bool)
        for label in face_labels:
            face_mask = face_mask | (parsing == label)

        # Generate hair mask
        hair_mask = (parsing == hair_label)

        return face_mask, hair_mask
    else:
        raise ValueError(f"Unsupported parsing model: {parse_model_name}")


def get_box_from_parsing_tensor(image_size, face_mask, hair_mask, target_image_size):
    """
    Calculate a bounding box around the areas marked by face_mask and hair_mask tensors,
    adjusting its size to be centered around the combined mask area. This box is
    then modified to ensure its size is around 5/4 of the target_image_size, either
    by padding or adjusting its dimensions, while making sure it doesn't exceed the
    image bounds. If the image size is smaller than 5/4 of the target_image_size,
    an exception is raised.

    Parameters:
    - image_size (tuple): The size of the image as (height, width).
    - face_mask (torch.Tensor): A binary mask tensor indicating the face location.
    - hair_mask (torch.Tensor): A binary mask tensor indicating the hair location.
    - target_image_size (tuple): The target image size as (height, width).

    Returns:
    - tuple: A tuple (new_rmin, new_rmax, new_cmin, new_cmax) representing the bounding
      box around the combined mask area, adjusted according to the specifications.

    Raises:
    - ValueError: If the image size is smaller than 5/4 of the target_image_size.
    """
    # Combine the masks
    combined_mask = face_mask | hair_mask
    
    # Find the bounding box
    rows = torch.any(combined_mask, dim=1)
    cols = torch.any(combined_mask, dim=0)
    rmin, rmax = torch.where(rows)[0][[0, -1]]
    cmin, cmax = torch.where(cols)[0][[0, -1]]
    
    # Adjust for tensor indexing and PyTorch operations
    rmin, rmax, cmin, cmax = rmin.item(), rmax.item(), cmin.item(), cmax.item()
    
    # Calculate target padding size
    target_pad_height = target_image_size[0] * 5 / 4
    target_pad_width = target_image_size[1] * 5 / 4
    
    if image_size[0] < target_pad_height or image_size[1] < target_pad_width:
        raise ValueError("Image size is smaller than 5/4 of the target image size.")
    
    # Calculate new box dimensions
    new_height = max(rmax - rmin, target_image_size[0]) * 5 / 4
    new_width = max(cmax - cmin, target_image_size[1]) * 5 / 4
    
    # Ensure new box does not exceed image_size
    new_height = min(new_height, image_size[0])
    new_width = min(new_width, image_size[1])
    
    # Center the new box around the original bounding box
    center_r = (rmin + rmax) / 2
    center_c = (cmin + cmax) / 2
    
    new_rmin = max(0, int(center_r - new_height / 2))
    new_rmax = min(image_size[0], int(center_r + new_height / 2))
    new_cmin = max(0, int(center_c - new_width / 2))
    new_cmax = min(image_size[1], int(center_c + new_width / 2))
    
    return (new_rmin, new_rmax, new_cmin, new_cmax)


'''Class to apply all transformations, including the RandomCropResizePad with a bounding box'''
class ImageMaskTransforms:
    def __init__(self, image_size, bounding_box):
        self.image_size = image_size
        self.bounding_box = bounding_box
        self.to_tensor = transforms.ToTensor()
        self.color_jitter = transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.01)

    def transform(self, img, do_flip, resize_scale, margins, apply_color_jitter=False):
        top, left, bottom, right = margins
        if do_flip:
            img = ImageOps.mirror(img)
        img = img.crop((left, top, right, bottom))
        new_width = int((right - left) * resize_scale)
        new_height = int((bottom - top) * resize_scale)
        img = img.resize((new_width, new_height), Image.BILINEAR)

        padding_left = (self.image_size[0] - new_width) // 2
        padding_top = (self.image_size[1] - new_height) // 2
        padding_right = self.image_size[0] - new_width - padding_left
        padding_bottom = self.image_size[1] - new_height - padding_top

        # Check if padding is negative
        if padding_left < 0 or padding_top < 0 or padding_right < 0 or padding_bottom < 0:
            # Calculate the scale to fit the image within self.image_size
            scale_width = self.image_size[0] / new_width
            scale_height = self.image_size[1] / new_height
            scale = min(scale_width, scale_height)  # Choose the smaller scale to ensure the image fits within the size

            # Recalculate dimensions based on the new scale
            new_width = int(new_width * scale)
            new_height = int(new_height * scale)
            img = img.resize((new_width, new_height), Image.BILINEAR)

            # Center the image and calculate new padding (should be non-negative now)
            padding_left = (self.image_size[0] - new_width) // 2
            padding_top = (self.image_size[1] - new_height) // 2
            padding_right = self.image_size[0] - new_width - padding_left
            padding_bottom = self.image_size[1] - new_height - padding_top

        # Apply padding if necessary (now ensured to be non-negative)
        img = ImageOps.expand(img, border=(padding_left, padding_top, padding_right, padding_bottom))

        if apply_color_jitter:
            img = self.color_jitter(img)

        return img

    def get_transformation_params(self):
        do_flip = random.random() > 0.5
        resize_scale = random.uniform(0.7, 1.0)
        new_rmin, new_rmax, new_cmin, new_cmax = self.bounding_box
        margin_height = (new_rmax - new_rmin) * random.uniform(0, 0.1)
        margin_width = (new_cmax - new_cmin) * random.uniform(0, 0.1)
        top = new_rmin + margin_height
        left = new_cmin + margin_width
        bottom = new_rmax - margin_height
        right = new_cmax - margin_width
        margins = (top, left, bottom, right)
        return do_flip, resize_scale, margins

    def tensor_to_pil(self, tensor):
        return to_pil_image(tensor)

    def pil_to_tensor(self, pil_img):
        return to_tensor(pil_img)

    def __call__(self, img, mask1=None, mask2=None):
        do_flip, resize_scale, margins = self.get_transformation_params()

        # Convert the image tensor to PIL for processing
        img_pil_transformed = self.transform(img, do_flip, resize_scale, margins, apply_color_jitter=True)
        # Convert back to tensor after transformation
        img_tensor_transformed = self.pil_to_tensor(img_pil_transformed)

        combined_mask_tensor = torch.zeros_like(img_tensor_transformed[0], dtype=torch.bool)

        if mask1 is not None:
            mask1_pil = self.tensor_to_pil(mask1.float())  # Convert boolean mask tensor to float tensor before to PIL
            save_path = "transformed_image.jpg"
            mask1_pil.save(save_path)
            mask1_pil = self.transform(mask1_pil, do_flip, resize_scale, margins)
            mask1_tensor_transformed = self.pil_to_tensor(mask1_pil).bool()  # Convert back to boolean tensor
            combined_mask_tensor |= mask1_tensor_transformed.squeeze()

        if mask2 is not None:
            mask2_pil = self.tensor_to_pil(mask2.float())  # Same conversion as for mask1
            mask2_pil = self.transform(mask2_pil, do_flip, resize_scale, margins)
            mask2_tensor_transformed = self.pil_to_tensor(mask2_pil).bool()
            combined_mask_tensor |= mask2_tensor_transformed.squeeze()

        # Apply the combined mask to the image tensor
        img_tensor_transformed *= combined_mask_tensor.unsqueeze(0).repeat(3, 1, 1)  # Ensure mask applies to all channels

        # Save the transformed image for logging purposes
        save_path = "transformed_image.jpg"
        transformed_image_pil = self.tensor_to_pil(img_tensor_transformed)
        transformed_image_pil.save(save_path)

        # Print log message with details
        print(f"Transformation applied: {'Flip' if do_flip else 'No flip'}, Resize scale: {resize_scale}, Margins: {margins}")
        print(f"Transformed image saved at: {os.path.abspath(save_path)}")

        # Return transformed image and masks
        return img_tensor_transformed, mask1_tensor_transformed if mask1 is not None else None, mask2_tensor_transformed if mask2 is not None else None
