import torch
import os
from transformers import (
    Blip2ForConditionalGeneration, 
    Blip2Processor,
    FuyuForCausalLM, 
    FuyuProcessor,
    LlavaForConditionalGeneration, 
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    BitsAndBytesConfig
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [RTX 5080 16GB 최적화] 4비트 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# --- [로그 헬퍼 함수] ---
def smart_load(class_obj, model_id, **kwargs):
    print(f"   🔎 '{model_id}' 찾는 중...", end=" ")
    try:
        obj = class_obj.from_pretrained(model_id, local_files_only=True, **kwargs)
        print("✅ [Cache] 로컬 발견!")
        return obj
    except Exception:
        print("🌐 [Download] 캐시에 없음. 다운로드 시작...")
        obj = class_obj.from_pretrained(model_id, local_files_only=False, **kwargs)
        print("      -> 다운로드 완료.")
        return obj

def load_blip2_base():
    print("\n--- BLIP-2 (Base) 4-bit ---")
    model_id = "Salesforce/blip2-opt-2.7b"
    
    model = smart_load(
        Blip2ForConditionalGeneration, 
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    processor = smart_load(Blip2Processor, model_id)
    return model, processor

def load_fuyu():
    print("\n--- Fuyu-8B 4-bit ---")
    model_id = "adept/fuyu-8b"
    
    print(f"   🔎 '{model_id}' (4-bit) 시도...", end=" ")
    try:
        model = FuyuForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto", local_files_only=True
        )
        print("✅ [Cache] 성공")
    except Exception:
        print("⚠️ [Cache Miss or Error] 실패. 인터넷 다운로드 또는 FP16 시도...")
        try:
             model = FuyuForCausalLM.from_pretrained(
                model_id, quantization_config=bnb_config, device_map="auto", local_files_only=False
            )
        except:
            print("      -> 4-bit 실패, FP16으로 다운로드/로드...")
            model = FuyuForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="auto"
            )
            
    processor = smart_load(FuyuProcessor, model_id)
    return model, processor

def load_llava13():
    print("\n--- LLaVA-13B 4-bit ---")
    model_id = "llava-hf/llava-1.5-13b-hf"
    
    model = smart_load(
        LlavaForConditionalGeneration, 
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    processor = smart_load(AutoProcessor, model_id)
    return model, processor

def load_clip_vit_l():
    print("\n--- CLIP-Large ---")
    model_id = "openai/clip-vit-large-patch14"
    
    print(f"   🔎 '{model_id}' 찾는 중...", end=" ")
    try:
        model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(DEVICE)
        print("✅ [Cache] 성공")
    except:
        print("🌐 [Download] 다운로드...")
        model = CLIPModel.from_pretrained(model_id, local_files_only=False).to(DEVICE)
        
    processor = smart_load(CLIPProcessor, model_id)
    return model, processor