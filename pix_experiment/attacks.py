import torch
import torch.nn as nn
import random
from attack_utils import DEVICE 
import torchvision.transforms.functional as TF
from PIL import Image
import io

class BPDA_JPEG_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, quality):
        pil_image = TF.to_pil_image(input_tensor.cpu().detach())
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)
        pil_image_compressed = Image.open(buffer)
        tensor_compressed = TF.to_tensor(pil_image_compressed).to(input_tensor.device)
        return tensor_compressed

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None 

def bpda_jpeg(tensor_image, quality=75):
    return BPDA_JPEG_Function.apply(tensor_image, quality)

def gaussian_blur(tensor_image, kernel_size=3):
    return TF.gaussian_blur(tensor_image, kernel_size=kernel_size)

def gaussian_noise(tensor_image, std=0.05):
    noise = torch.randn_like(tensor_image) * std
    return tensor_image + noise

eot_transforms = [
    lambda x: bpda_jpeg(x, quality=random.randint(70, 95)), 
    lambda x: gaussian_blur(x, kernel_size=3),
    lambda x: gaussian_noise(x, std=random.uniform(0.01, 0.05)),
    lambda x: x
]

def pgd_attack(models, labels, original_tensor, epsilon, alpha, num_iter, seed, eot_samples=10):
    
    from attack_utils import evaluate_image_tensor

    original_label, _ = evaluate_image_tensor(original_tensor, models[0], labels)
    try:
        original_label_idx = [int(k) for k, v in labels.items() if v == original_label][0]
    except (IndexError, KeyError): # KeyError 추가
        print(f"경고: 원본 라벨 '{original_label}'을 라벨 목록에서 찾을 수 없습니다. 랜덤 타겟을 설정합니다.")
        original_label_idx = -1

    random.seed(seed + (original_label_idx if original_label_idx != -1 else 0))
    num_classes = len(labels)
    
    while True:
        target_label_idx = random.randint(0, num_classes - 1)
        if target_label_idx != original_label_idx:
            break
    target_label = labels[str(target_label_idx)]
    
    target = torch.tensor([target_label_idx]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    print(f" -> BPDA 기반 EOT 공격 목표 설정 완료: {target_label} (원본: {original_label})")

    perturbed_tensor = original_tensor.clone().detach().to(DEVICE)
    
    from tqdm import tqdm
    for i in tqdm(range(num_iter), desc="PGD Attack"):
        perturbed_tensor.requires_grad = True
        
        total_eot_loss = 0
        for _ in range(eot_samples):
            transform = random.choice(eot_transforms)
            transformed_image = transform(perturbed_tensor)
            
            for model in models:
                output = model(transformed_image.unsqueeze(0))
                total_eot_loss += loss_fn(output, target)
        
        avg_loss = total_eot_loss / (eot_samples * len(models))

        for model in models:
            model.zero_grad()
            
        avg_loss.backward()
        
        attack_update = alpha * perturbed_tensor.grad.sign()
        perturbed_tensor = perturbed_tensor.detach() - attack_update
        eta = torch.clamp(perturbed_tensor - original_tensor.to(DEVICE), -epsilon, epsilon)
        perturbed_tensor = (original_tensor.to(DEVICE) + eta).detach()
        
    print(f" -> BPDA-EOT-PGD 공격 완료 ({num_iter}회 반복, 샘플링 {eot_samples}회)")
    return perturbed_tensor.cpu()