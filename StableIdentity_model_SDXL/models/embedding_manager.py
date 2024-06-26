import torch
from torch import nn
from einops import rearrange
import numpy as np
from typing import List
from StableIdentity_model_SDXL.models.id_embedding.helpers import get_rep_pos, shift_tensor_dim0
from StableIdentity_model_SDXL.models.id_embedding.meta_net import StyleVectorizer
from functools import partial
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from StableIdentity_model_SDXL.models.celeb_embeddings import _get_celeb_embeddings_basis


DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, return_length=True,
                               return_overflowing_tokens=False, return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    # assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens




def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))



class EmbeddingManagerId_adain(nn.Module):
    def __init__(
            self,
            tokenizer,
            tokenizer_2,
            text_encoder,
            text_encoder_2,
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),            
            num_embeds_per_token: int = 2,
            mlp_depth: int = 2,
            token_dim: int = 768,
            token_dim_2: int = 1280,
            vit_out_dim: int = 512,
            **kwargs
    ):
        super().__init__()
        self.device = device
        self.num_es = num_embeds_per_token

        self.get_token_for_string = partial(get_clip_token_for_string, tokenizer)        
        self.get_embedding_for_tkn = partial(get_embedding_for_clip_token, text_encoder.text_model.embeddings)  

        self.token_dim = token_dim
        self.token_dim_2 = token_dim_2

        ''' 1. Placeholder mapping dicts '''
        self.placeholder_token = self.get_token_for_string("*")[0][1]   


        ''' id embedding '''
        self.celeb_embeddings_mean, self.celeb_embeddings_std = _get_celeb_embeddings_basis(tokenizer, text_encoder, "StableIdentity_model_SDXL/datasets_face/selected_celeb_names.txt")
        
        self.celeb_embeddings_mean = self.celeb_embeddings_mean.to(device)
        self.celeb_embeddings_std = self.celeb_embeddings_std.to(device)
        
        self.celeb_embeddings_mean_2, self.celeb_embeddings_std_2 = _get_celeb_embeddings_basis(tokenizer_2, text_encoder_2, "StableIdentity_model_SDXL/datasets_face/selected_celeb_names.txt")
        
        self.celeb_embeddings_mean_2 = self.celeb_embeddings_mean_2.to(device)
        self.celeb_embeddings_std_2 = self.celeb_embeddings_std_2.to(device)

        self.face_projection_layer = StyleVectorizer(vit_out_dim, token_dim * self.num_es,
                                    depth=mlp_depth, lr_mul=0.1)
        
        self.face_projection_layer_2 = StyleVectorizer(vit_out_dim, token_dim_2 * self.num_es,
                                    depth=mlp_depth, lr_mul=0.1)

        

    def forward(
            self,
            tokenized_text,  
            embedded_text,  
            tokenizer_id,
            face_img_embeddings = None,
            timesteps = None,
    ):
        batch_size, n, device = *tokenized_text.shape, tokenized_text.device
        
        if face_img_embeddings is not None:
            if tokenizer_id == 1:
                residual_embedding = self.face_projection_layer(face_img_embeddings).view(batch_size, self.num_es, self.token_dim)
                text_img_embeddings = self.celeb_embeddings_mean + residual_embedding * self.celeb_embeddings_std
            elif tokenizer_id == 2:
                residual_embedding = self.face_projection_layer_2(face_img_embeddings).view(batch_size, self.num_es, self.token_dim_2)
                text_img_embeddings = self.celeb_embeddings_mean_2 + residual_embedding * self.celeb_embeddings_std_2
                
        placeholder_pos = get_rep_pos(tokenized_text, [self.placeholder_token])
        placeholder_pos = np.array(placeholder_pos)
        if len(placeholder_pos) != 0:
            embedded_text[placeholder_pos[:, 0], placeholder_pos[:, 1]] = text_img_embeddings[:, 0]
            embedded_text[placeholder_pos[:, 0], placeholder_pos[:, 1] + 1] = text_img_embeddings[:, 1]

        return embedded_text



    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cuda')
        
        if ckpt.get("face_projection_layer") is not None:
            self.face_projection_layer = ckpt.get("face_projection_layer").float()
            
        print('[Embedding Manager] weights loaded.')


    def save(self, face_img_embeddings, emb_path, tokenizer_id):
        if tokenizer_id == 1:
            # save embeddings
            residual_embedding = self.face_projection_layer(face_img_embeddings).view(face_img_embeddings.shape[0], self.num_es, self.token_dim)
            text_img_embeddings = self.celeb_embeddings_mean + residual_embedding * self.celeb_embeddings_std
            torch.save(text_img_embeddings, emb_path)
        elif tokenizer_id == 2:
            residual_embedding = self.face_projection_layer_2(face_img_embeddings).view(face_img_embeddings.shape[0], self.num_es, self.token_dim_2)
            text_img_embeddings = self.celeb_embeddings_mean_2 + residual_embedding * self.celeb_embeddings_std_2
            torch.save(text_img_embeddings, emb_path)
        else:
            raise ValueError(f"Wrong Tokenizer ID {tokenizer_id}")

    def trainable_projection_parameters(self):
        trainable_list = []
        trainable_list.extend(list(self.face_projection_layer.parameters())) 
        trainable_list.extend(list(self.face_projection_layer_2.parameters())) 
        return trainable_list