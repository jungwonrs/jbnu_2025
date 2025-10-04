import os
import torch
from attack_utils import (
    preprocess_image, evaluate_image_tensor, 
    calculate_image_metrics, setup_model_and_labels, DEVICE,
    jpeg_compress, gaussian_blur, gaussian_noise
)
from secret_key_util import extract_secret

defenses = {
    "jpeg_90": lambda x: jpeg_compress(x, quality=90),
    "jpeg_70": lambda x: jpeg_compress(x, quality=70),
    "blur": lambda x: gaussian_blur(x, kernel_size=3),
    "noise": lambda x: gaussian_noise(x, std=0.05)
}

def evaluate_performance(original_tensor, adversarial_tensor, delta_tensor, pattern_key,
                         secret_key_strength, evaluation_model_names):
    attacked_with_key_tensor = adversarial_tensor + (pattern_key.cpu() * secret_key_strength)
    restored_adv_tensor = extract_secret(attacked_with_key_tensor, pattern_key, secret_key_strength).cpu()
    fully_restored_tensor = restored_adv_tensor - delta_tensor
    
    results = {}
    
    psnr_attack, ssim_attack = calculate_image_metrics(original_tensor, restored_adv_tensor)
    psnr_restore, ssim_restore = calculate_image_metrics(original_tensor, fully_restored_tensor)
    results['PSNR_attack'] = round(psnr_attack, 2)
    results['SSIM_attack'] = round(ssim_attack, 4)
    results['PSNR_restored'] = round(psnr_restore, 2)
    results['SSIM_restored'] = round(ssim_restore, 4)

    print(f" -> 품질 평가 완료: PSNR={psnr_attack:.2f}, SSIM={ssim_attack:.4f}")
    
    for model_name in evaluation_model_names:
        model, labels = setup_model_and_labels(model_name)
        
        original_label, _ = evaluate_image_tensor(original_tensor, model, labels)
        attacked_label, _ = evaluate_image_tensor(restored_adv_tensor, model, labels)
        
        results[f'{model_name}_attack_success_clean'] = (original_label != attacked_label)
        results[f'{model_name}_original_label'] = original_label
        results[f'{model_name}_attacked_label_clean'] = attacked_label
        
        for defense_name, transform in defenses.items():
            defended_tensor = transform(restored_adv_tensor.cpu()).to(DEVICE)
            defended_label, _ = evaluate_image_tensor(defended_tensor, model, labels)
            results[f'{model_name}_attack_success_after_{defense_name}'] = (original_label != defended_label)

    print(f" -> 강인성 평가 완료.")
    return results