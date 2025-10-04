import torch
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
KEY_FILE = "secret.key"

def generate_pattern_from_seed(seed, size=(3, 224, 224)): 
    torch.manual_seed(seed)
    pattern_key = torch.randn(size)
    return pattern_key.to(DEVICE)

def embed_secret (image_tensor, key_tensor, strength=0.2):
    return image_tensor.to(DEVICE) + (key_tensor * strength)

def extract_secret (image_with_key_tensor, key_tensor, strength=0.2):
    return image_with_key_tensor.to(DEVICE) - (key_tensor * strength)