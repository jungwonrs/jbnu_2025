# run.py

import os
import torch
import pandas as pd
from PIL import Image
from itertools import product
import hashlib # ⭐ 수정: 누락된 hashlib 임포트

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
    "num_images_to_test": [1], # 0 이면 모든 이미지 다 쓸듯?
    "attack_model_names": [['resnet50', 'vgg16']],

    # --- PGD 하이퍼파라미터 ---
    "epsilon": [4/255, 8/255],  # 총 변화량 한계. 공격의 '강도'와 '은밀함'을 결정하는 가장 중요한 값.
                                # 값이 클수록 공격은 강해지지만 이미지 왜곡이 눈에 띌 수 있습니다. (예: 8/255 > 4/255)

    "alpha": [2/255],           # 스텝 크기 (보폭). 한 번의 반복에서 이미지를 얼마나 크게 수정할지 결정합니다.
                                # 공격의 '수렴 속도'와 '안정성'에 영향을 줍니다. 보통 epsilon보다 작은 값을 사용합니다.

    "num_iter": [20, 40],       # 공격 반복 횟수. AI를 속이기 위해 이미지를 수정하는 과정을 몇 번 반복할지 결정합니다.
                                # 값이 클수록 공격은 더 정교하고 강력해지지만, 생성 시간이 오래 걸립니다.

    # --- EOT 하이퍼파라미터 ---
    "eot_samples": [10],        # EOT 샘플링 횟수. 강인함(Robustness)을 기르기 위해 매 스텝마다 몇 개의 랜덤 왜곡을 적용해볼지 결정합니다.
                                # 값이 클수록 JPEG 압축, 노이즈 등에 강인한 공격이 만들어지지만, 생성 속도가 매우 느려집니다.

    # 3. 비밀 키 파라미터
    "secret_key_strength": [0.1], # 비밀 키 적용 강도. 최종 이미지에 비밀 키 패턴을 얼마나 '진하게' 덧씌울지 결정합니다.
                                 # 값이 크면 비밀 키의 보안성(내구성)은 높아지나, 이미지 품질(PSNR/SSIM)이 저하될 수 있습니다.

    "evaluation_model_names": [[
        'resnet50',          # 표준적인 현대 CNN
        'vgg16',             # 단순하고 깊은 고전적 CNN
        'mobilenet_v2',      # 모바일 기기를 위한 경량 CNN
        'efficientnet_b0',   # 효율적으로 설계된 최신 CNN
        'convnext_tiny',     # Transformer에 영감을 받은 최신 CNN
        'vit_b_16'           # Vision Transformer (비-CNN 계열)
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
            # (디버깅 편의를 위해 이미지 저장은 유지)
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