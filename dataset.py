import random
from torch.utils.data import Dataset
from torch import nn
from torchvision.transforms import transforms
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, ViTImageProcessor, ViTModel
from PIL import Image
from util import get_mask_from_parsing, get_box_from_parsing_tensor, ImageMaskTransforms
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import os

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]

class FaceDataset(Dataset):
    def __init__(self,
                 face_img_path,
                 device,
                 tar_image_size=512, #512
                 face_parser_model_path="jonathandinu/face-parsing",
                 vit_model_path="jayanta/google-vit-base-patch16-224-face",
                 augment_len=1000,
                 dtype=torch.float32,):
        super(FaceDataset, self).__init__()
        
        self.device = device
        self.dtype = dtype
        self.face_img_path = face_img_path
        
        if tar_image_size == 256:
            self.mask_size = 32
        elif tar_image_size == 512:
            self.mask_size = 64

        '''Load the face parser model'''
        '''Fine-tune the face parser model from SegFormer (b5-sized)'''
        face_parser_processor = SegformerImageProcessor.from_pretrained(face_parser_model_path)
        face_parser_model = SegformerForSemanticSegmentation.from_pretrained(face_parser_model_path)
        face_parser_model.to(device)

        '''Load the VIT model'''
        '''Fine-tune the VIT model from Google Research (base-sized) for face recognition'''
        self.vit_face_recog_processor = ViTImageProcessor.from_pretrained(vit_model_path)
        self.vit_face_recognition_model = ViTModel.from_pretrained(vit_model_path)
        self.vit_face_recognition_model.to(device)

        '''Load the face images'''
        self.face_img = Image.open(face_img_path)
        self.length = augment_len
        self.tar_img_size = tar_image_size

        '''Face and Hair Mask'''
        inputs = face_parser_processor(images=self.face_img, return_tensors="pt").to(device)
        outputs = face_parser_model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

        # resize output to match input image dimensions
        upsampled_logits = nn.functional.interpolate(logits,
                        size=self.face_img.size[::-1], # H x W
                        mode='bilinear',
                        align_corners=False)
        parsed_labels = upsampled_logits.argmax(dim=1)[0]
        self.face_mask, self.hair_mask = get_mask_from_parsing(parsed_labels, parse_model_name=face_parser_model_path)
        bounding_box = get_box_from_parsing_tensor(self.face_img.size, self.face_mask, self.hair_mask, (tar_image_size, tar_image_size))

        '''Transform the face images'''
        self.transformation = ImageMaskTransforms(image_size=(self.tar_img_size, self.tar_img_size), bounding_box=bounding_box)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''Augment the face images'''
        cur_img, cur_mask_face, cur_mask_hair = self.transformation(self.face_img, self.face_mask, self.hair_mask)
        
        face_mask_resize = F.interpolate(cur_mask_face.unsqueeze(0).float(), (self.mask_size, self.mask_size), mode='bicubic', align_corners=True)[0]
        face_mask_resize[face_mask_resize >= 0.5] = 1
        face_mask_resize[face_mask_resize < 0.5] = 0
        
        hair_mask_resize = F.interpolate(cur_mask_hair.unsqueeze(0).float(), (self.mask_size, self.mask_size), mode='bicubic', align_corners=True)[0]
        hair_mask_resize[hair_mask_resize >= 0.5] = 1
        hair_mask_resize[hair_mask_resize < 0.5] = 0
        
        vit_input = self.vit_face_recog_processor(images=cur_img, return_tensors="pt")["pixel_values"][0]
        vit_input = vit_input.to(self.device)
        
        # with autocast():
        vit_cls_output = self.vit_face_recognition_model(vit_input.unsqueeze(0).to(self.device)).last_hidden_state[:, 0]

        placeholder_string = "*"
        text = random.choice(imagenet_templates_small).format('%s person' % placeholder_string)
        item = {}
        item["vit_output"] = vit_cls_output
        item["text"] = text
        item["mask_face"] = face_mask_resize
        item["mask_hair"] = hair_mask_resize
        item["face_img"] = cur_img
        return item
    
    def get_vit_cls_output(self):
        face_img, _, _ = self.transformation(self.face_img, self.face_mask, self.hair_mask)
        vit_input = self.vit_face_recog_processor(images=face_img, return_tensors="pt")["pixel_values"][0]
        vit_input = vit_input.to(self.device)
        vit_cls_output = self.vit_face_recognition_model(vit_input.unsqueeze(0).to(self.device)).last_hidden_state[:, 0]
        return vit_cls_output
    
    def get_img_name(self):
        return os.path.basename(self.face_img_path)[:-4]