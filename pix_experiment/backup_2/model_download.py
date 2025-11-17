import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import snapshot_download
from call_target_model import classifier_model 
import torch

HF_TOKEN = "-" 

model_id_list = [
    # --- 소스 모델 ---
    "llava-hf/llava-1.5-7b-hf",
    "Salesforce/instructblip-flan-t5-xl",
    "openai/clip-vit-base-patch32",
    
    # 2. (필수) 존재하지 않는 ID -> 표준 ViT 모델 ID로 수정
    "google/vit-base-patch16-224", 
    
    # --- 전이 모델 ---
    "Qwen/Qwen-VL-Chat",
    "Salesforce/blip2-opt-2.7b",
    "adept/fuyu-8b",
    "llava-hf/llava-1.5-13b-hf",
    "openai/clip-vit-large-patch14"
]

def download_all_models():
    
    # --- Hugging Face ---
    print("="*30)
    print("Hugging Face 모델 다운로드를 시작합니다...")
    print("="*30)
    for model_id in model_id_list:
        print(f"\n--- {model_id} 다운로드 중 ---")
        try:
            snapshot_download(
                repo_id=model_id,
                token=HF_TOKEN
            )
            print(f"--- {model_id} 다운로드 완료 ---")
        except Exception as e:
            print(f"!!! {model_id} 다운로드 실패: {e}")
            if "Gated" in str(e):
                print(f"!!! {model_id} 모델은 웹사이트에서 '약관 동의'가 필요할 수 있습니다.")

    
    # --- Torchvision  ---
    print("\n" + "="*30)
    print("Torchvision 모델 다운로드를 시작합니다...")
    print("="*30)
    try:
        print("--- ResNet-50 다운로드 중 ---")
        _ = classifier_model.resnet()
        print("--- VGG16 다운로드 중 ---")
        _ = classifier_model.vgg()
        print("--- Torchvision 모델 다운로드 완료 ---")
    except Exception as e:
        print(f"!!! Torchvision 모델 다운로드 실패: {e}")

    print("\n" + "*"*30)
    print("모든 모델 파일 다운로드가 완료되었습니다.")
    print("이제 '실행기' 스크립트에서 '로더'를 호출해 사용하세요.")
    print("*"*30)

if __name__ == "__main__":
    download_all_models()