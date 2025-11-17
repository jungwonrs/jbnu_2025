import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import json
import os
import math
from config import DEVICE, COCO_ANNOTATIONS, COCO_VAL_IMAGES

def ssim_loss(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = img1.mean(dim=[1, 2])
    mu2 = img2.mean(dim=[1, 2])
    sigma1_sq = img1.var(dim=[1, 2])
    sigma2_sq = img2.var(dim=[1, 2])
    sigma12 = ((img1 - mu1.view(-1, 1, 1)) * (img2 - mu2.view(-1, 1, 1))).mean(dim=[1, 2])
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def format_params(params, limit=20):
    cols = {}
    for i, (k, v) in enumerate(params.items()):
        val_str = str(v)
        if len(val_str) > limit:
            val_str = val_str[:limit] + "..."
        cols[f"algo_para_{i+1}"] = f"{k}={val_str}"
    return cols

def no_attack(img_tensor):
    return img_tensor

def jpeg_attack(img_tensor, quality=90):
    if img_tensor.dim() == 4: img_tensor = img_tensor.squeeze(0)
    pil_img = transforms.ToPILImage()(img_tensor.cpu())
    import io
    buffer = io.BytesIO()
    pil_img.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    jpeg_pil = Image.open(buffer)
    return transforms.ToTensor()(jpeg_pil).unsqueeze(0).to(DEVICE)

def blur_attack(img_tensor, sigma=1.0):
    return transforms.GaussianBlur(kernel_size=5, sigma=sigma)(img_tensor)

def noise_attack(img_tensor, std=0.01):
    noise = torch.randn_like(img_tensor) * std
    return torch.clamp(img_tensor + noise, 0, 1)

def calculate_metrics(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr = 100.0
        ssim_val = 1.0
    else:
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
        ssim_val = ssim_loss(img1.squeeze(0), img2.squeeze(0))
    return psnr, ssim_val

def load_coco_data():
    if not os.path.exists(COCO_ANNOTATIONS):
        print(f"!!! Error: 파일 없음 {COCO_ANNOTATIONS}")
        return [], {}
    try:
        with open(COCO_ANNOTATIONS, 'r') as f: data = json.load(f)
    except Exception as e:
        print(f"Error loading COCO: {e}")
        return [], {}

    category_map = {cate['id']: cate['name'] for cate in data['categories']}
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns: img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann['category_id'])
    
    coco_data_list = []
    for img_info in data['images']:
        img_id = img_info['id']
        full_path = os.path.join(COCO_VAL_IMAGES, img_info['file_name'])
        if not os.path.exists(full_path): continue
        if img_id in img_to_anns and len(img_to_anns[img_id]) > 0:
            cat_id = img_to_anns[img_id][0]
            coco_data_list.append((full_path, cat_id, category_map.get(cat_id, "Unknown")))
    return coco_data_list, category_map

def get_classifier_prediction(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        pred_id = logits.argmax(dim=1).item()
    return pred_id

def generate_attack_jobs(config_list):
    jobs = []
    for conf in config_list:
        if "epsilon" in conf["param_grid"]:
            for eps in conf["param_grid"]["epsilon"]:
                jobs.append({"name": conf["name"], "func": conf["func"], "params": {"epsilon": eps}})
    return jobs

def get_vlm_prediction(model, processor, image, gt_label=None, model_name=""):
    m_name = str(model_name)
    
    if "InstructBLIP" in m_name or "BLIP" in m_name:
        prompt = "Question: Describe this image in detail. Answer:"
    elif "LLaVA" in m_name:
        prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"
    elif "Fuyu" in m_name:
        prompt = "Generate a representation of this image.\n"
    else:
        prompt = "Describe this image."

    device = model.device if hasattr(model, "device") else DEVICE
    
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        # [수정] 200토큰으로 넉넉하게!
        max_t = 200
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_t, 
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id if hasattr(processor, "tokenizer") else None
        )
    
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    if "ASSISTANT:" in decoded:
        decoded = decoded.split("ASSISTANT:")[-1]
    elif "Answer:" in decoded:
        decoded = decoded.split("Answer:")[-1]
    elif prompt in decoded:
        decoded = decoded.replace(prompt, "")
        
    return decoded.strip()