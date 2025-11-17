import torch
import os
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration, 
    InstructBlipForConditionalGeneration, 
    InstructBlipProcessor,
    CLIPModel,
    CLIPProcessor,
    ViTForImageClassification,
    ViTImageProcessor,
    BitsAndBytesConfig
)
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [RTX 5080 16GB 최적화] 4비트 로딩 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# --- [로그 헬퍼 함수] ---
def smart_load(class_obj, model_id, **kwargs):
    """
    1. 로컬 캐시(local_files_only=True)로 먼저 시도
    2. 실패하면 인터넷(local_files_only=False)으로 시도
    하며 로그를 출력함.
    """
    print(f"   🔎 '{model_id}' 찾는 중...", end=" ")
    try:
        # 1. 캐시 로드 시도
        obj = class_obj.from_pretrained(model_id, local_files_only=True, **kwargs)
        print("✅ [Cache] 로컬 발견! (인터넷 X)")
        return obj
    except Exception:
        # 2. 실패 시 다운로드
        print("🌐 [Download] 캐시에 없음. 다운로드 시작...")
        obj = class_obj.from_pretrained(model_id, local_files_only=False, **kwargs)
        print("      -> 다운로드 및 로드 완료.")
        return obj

class vml_model:
    @staticmethod
    def llava7():
        print("\n--- LLaVA-1.5-7B (4-bit) ---")
        model_id = "llava-hf/llava-1.5-7b-hf"
        
        model = smart_load(
            LlavaForConditionalGeneration, 
            model_id, 
            quantization_config=bnb_config, 
            device_map="auto"
        )
        processor = smart_load(AutoProcessor, model_id)
        return model, processor

    @staticmethod
    def instructblip():
        print("\n--- InstructBLIP (4-bit) ---")
        model_id = "Salesforce/instructblip-flan-t5-xl"
        
        model = smart_load(
            InstructBlipForConditionalGeneration, 
            model_id, 
            quantization_config=bnb_config, 
            device_map="auto"
        )
        processor = smart_load(InstructBlipProcessor, model_id)
        return model, processor

    @staticmethod
    def clip_vit_b():
        print("\n--- CLIP (ViT-B/32) ---")
        model_id = "openai/clip-vit-base-patch32"
        
        # CLIP은 모델 크기가 작으므로 4비트 없이 바로 GPU로 로드
        print(f"   🔎 '{model_id}' (No-Quant) 찾는 중...", end=" ")
        try:
            model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(DEVICE)
            print("✅ [Cache] 성공")
        except:
            print("🌐 [Download] 다운로드...")
            model = CLIPModel.from_pretrained(model_id, local_files_only=False).to(DEVICE)
            
        processor = smart_load(CLIPProcessor, model_id)
        return model, processor


class classifier_model:
    @staticmethod
    def vit():
        print("\n--- ViT-B/32 ---")
        model_id = "google/vit-base-patch16-224"
        
        print(f"   🔎 '{model_id}' 찾는 중...", end=" ")
        try:
            model = ViTForImageClassification.from_pretrained(model_id, local_files_only=True).to(DEVICE)
            print("✅ [Cache] 성공")
        except:
            print("🌐 [Download] 다운로드...")
            model = ViTForImageClassification.from_pretrained(model_id, local_files_only=False).to(DEVICE)
            
        processor = smart_load(ViTImageProcessor, model_id)
        return model, processor

    @staticmethod
    def resnet():
        print("\n--- ResNet-50 ---")
        print("   ✅ [Built-in] Torchvision 내장 모델 사용")
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights).to(DEVICE)
        model.eval()
        processor = weights.transforms()
        return model, processor

    @staticmethod
    def vgg():
        print("\n--- VGG16 ---")
        print("   ✅ [Built-in] Torchvision 내장 모델 사용")
        weights = VGG16_Weights.IMAGENET1K_V1
        model = vgg16(weights=weights).to(DEVICE)
        model.eval()
        processor = weights.transforms()
        return model, processor