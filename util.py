import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
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
class ImageTransforms:
    def __init__(self, image_size, bounding_box, pad_value=(0, 0, 0)):
        self.image_size = image_size
        self.bounding_box = bounding_box
        self.pad_value = pad_value
        self.pre_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.01),
        ])
        self.post_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def random_crop_resize_pad(self, img):
        new_rmin, new_rmax, new_cmin, new_cmax = self.bounding_box
        margin_height = (new_rmax - new_rmin) * random.uniform(0, 0.1)
        margin_width = (new_cmax - new_cmin) * random.uniform(0, 0.1)
        top = new_rmin + margin_height
        left = new_cmin + margin_width
        bottom = new_rmax - margin_height
        right = new_cmax - margin_width
        img = img.crop((left, top, right, bottom))
        resize_scale = random.uniform(0.5, 1.0)
        new_width = int((right - left) * resize_scale)
        new_height = int((bottom - top) * resize_scale)
        img = img.resize((new_width, new_height), Image.BILINEAR)
        padding_left = (self.image_size[0] - new_width) // 2
        padding_top = (self.image_size[1] - new_height) // 2
        padding_right = self.image_size[0] - new_width - padding_left
        padding_bottom = self.image_size[1] - new_height - padding_top
        img = ImageOps.expand(img, border=(padding_left, padding_top, padding_right, padding_bottom), fill=self.pad_value)
        return img

    def __call__(self, img):
        img = self.pre_transforms(img)
        img = self.random_crop_resize_pad(img)
        img = self.post_transforms(img)
        return img
