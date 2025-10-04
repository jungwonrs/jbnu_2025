# run.py

import os
import torch
import pandas as pd
from PIL import Image
from itertools import product
import hashlib

# --- 유틸리티 및 모듈 임포트 ---
from attack_utils import setup_model_and_labels, preprocess_image, save_tensor_as_image
from blockchain_utils import generate_private_key, private_key_to_seed
from secret_key_util import generate_pattern_from_seed, embed_secret
from attacks import pgd_attack
from evaluation import evaluate_performance

# ==============================================================================
# 실험 설정
# ==============================================================================
CONFIG = {
    "input_dir": ["input_images"],
    "output_dir": ["output_images"],
    "num_images_to_test": [1],
    "attack_model_names": [['resnet50', 'vgg16']],
    "epsilon": [8/255],
    "alpha": [2/255],
    "num_iter": [20],
    "eot_samples": [10],
    "secret_key_strength": [0.1],

    "evaluation_model_names": [[
        # --- 기존 목록 ---
        'resnet50',          # 표준적인 현대 CNN
        'vgg16',             # 단순하고 깊은 고전적 CNN
        'mobilenet_v2',      # 모바일 기기를 위한 경량 CNN
        'efficientnet_b0',   # 효율적으로 설계된 최신 CNN
        'convnext_tiny',     # Transformer에 영감을 받은 최신 CNN
        'vit_b_16',          # Vision Transformer (비-CNN 계열)
        
        # --- 🚀 추가된 강력한 CNN 계열 ---
        'resnext50_32x4d',   # ResNet의 성능을 높인 확장판
        'wide_resnet50_2',   # ResNet을 깊게 대신 넓게 만든 변형
        
        # --- 🧩 추가된 독특한 구조의 CNN 계열 ---
        'densenet121',       # 특징 재사용을 극대화한 밀집 연결 구조
        'inception_v3',      # 다양한 스케일의 특징을 동시에 분석
        
        # --- 🤖 추가된 최신 Transformer 계열 (timm 라이브러리 필요) ---
        'swin_tiny_patch4_window7_224', # Swin Transformer
        'deit_base_patch16_224'         # DeiT (Data-efficient Image Transformer)
    ]],
}
RESULTS_EXCEL_PATH = "experiment_results_summary.xlsx"
# ==============================================================================

def main():
    keys, values = zip(*CONFIG.items())
    experiments = [dict(zip(keys, v)) for v in product(*values)]
    print(f" 총 {len(experiments)}개의 실험을 시작합니다.")
    all_results = []

    for i, config in enumerate(experiments):
        print(f"\n{'='*30} 실험 {i+1}/{len(experiments)} 시작 {'='*30}")
        print(f" -> 현재 설정: Epsilon={config['epsilon']:.4f}, Iter={config['num_iter']}, EOT Samples={config['eot_samples']}")

        os.makedirs(config["input_dir"], exist_ok=True)
        os.makedirs(config["output_dir"], exist_ok=True)
        
        private_key = generate_private_key()
        seed = private_key_to_seed(private_key)
        pattern_key = generate_pattern_from_seed(seed)

        attack_models = []
        labels = None
        for name in config["attack_model_names"]:
            model, lbls = setup_model_and_labels(name)
            attack_models.append(model)
            if labels is None: labels = lbls

        all_image_files = [f for f in os.listdir(config["input_dir"]) if f.endswith(('png', 'jpg', 'jpeg'))]
        image_files_to_run = all_image_files[:config["num_images_to_test"]] if config["num_images_to_test"] > 0 else all_image_files

        if not image_files_to_run:
            print(" -> 입력 이미지가 없어 이 실험을 건너뜁니다.")
            continue
            
        for filename in image_files_to_run:
            print(f"\n--- {filename} 처리 중 ---")
            original_pil = Image.open(os.path.join(config["input_dir"], filename)).convert("RGB")
            original_tensor = preprocess_image(original_pil)
            
            adversarial_tensor = pgd_attack(
                models=attack_models, labels=labels, original_tensor=original_tensor,
                epsilon=config["epsilon"], alpha=config["alpha"], num_iter=config["num_iter"], seed=seed,
                eot_samples=config["eot_samples"]
            )
            
            delta_tensor = adversarial_tensor - original_tensor
            delta_hash = hashlib.sha256(delta_tensor.numpy().tobytes()).hexdigest()
            print(f" -> Delta Tensor 생성 완료. Hash: {delta_hash[:16]}...")
            
            final_image_tensor = embed_secret(adversarial_tensor, pattern_key.cpu(), config["secret_key_strength"])
            save_tensor_as_image(original_tensor, os.path.join(config["output_dir"], f"original_{filename}"))
            save_tensor_as_image(final_image_tensor, os.path.join(config["output_dir"], f"attacked_{filename}"))
            
            evaluation_results = evaluate_performance(
                original_tensor=original_tensor,
                adversarial_tensor=adversarial_tensor,
                delta_tensor=delta_tensor,
                pattern_key=pattern_key,
                secret_key_strength=config["secret_key_strength"],
                evaluation_model_names=config["evaluation_model_names"]
            )
            
            evaluation_results['filename'] = filename
            evaluation_results['delta_hash'] = delta_hash
            for key, value in config.items():
                evaluation_results[key] = str(value) if isinstance(value, list) else value

            all_results.append(evaluation_results)

    if all_results:
        df = pd.DataFrame(all_results)
        param_cols = list(CONFIG.keys()) + ['delta_hash']
        other_cols = [col for col in df.columns if col not in param_cols and col != 'filename']
        df = df[['filename'] + param_cols + other_cols]
        
        df.to_excel(RESULTS_EXCEL_PATH, index=False, engine='openpyxl')
        print(f"\n\n✅ 모든 실험 완료! 최종 결과가 '{RESULTS_EXCEL_PATH}'에 저장되었습니다.")

if __name__ == '__main__':
    main()