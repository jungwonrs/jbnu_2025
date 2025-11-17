import torch
from call_target_model import vml_model, classifier_model
from call_transfer_model import (
    load_blip2_base,
    load_fuyu,
    load_llava13,
    load_clip_vit_l  
)

all_loader_functions = [
    # --- 소스 (Target) ---
    vml_model.llava7,         # 1
    vml_model.instructblip,   # 2
    vml_model.clip_vit_b,     # 3
    classifier_model.vit,     # 4
    classifier_model.resnet,  # 5
    classifier_model.vgg,     # 6
    
    # --- 전이 (Transfer) ---
    load_blip2_base,          # 7
    load_fuyu,                # 8
    load_llava13,             # 9
    load_clip_vit_l           # 10
]


def test_all_models_sequentially():
    print("="*30)
    print("전체 11개 모델 순차 로드 테스트 시작...")
    print("="*30)
    
    success_count = 0
    failed_models = []

    for i, loader_func in enumerate(all_loader_functions):
        model_name = loader_func.__name__
        if hasattr(loader_func, '__self__'): 
            model_name = f"{loader_func.__self__.__name__}.{model_name}"

        print(f"\n--- [{i+1}/{len(all_loader_functions)}] {model_name} 로드 테스트 시작 ---")
        
        try:
            # --- 모델 로드 (VRAM 사용 시작) ---
            model, processor = loader_func()
            
            print(f"✅ [성공] {model_name} 로드 완료 (Model: {type(model)})")
            success_count += 1
            
            # --- (필수) VRAM에서 즉시 삭제 ---
            print(f"--- {model_name} VRAM에서 삭제 중 ---")
            del model
            del processor
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache() 
                
            print(f"--- {model_name} VRAM 정리 완료 ---\n")

        except Exception as e:
            print(f"❌ [실패] {model_name} 로드 중 오류 발생: {e}")
            failed_models.append(model_name)
            
            # CUDA 오류가 났을 경우, 캐시를 비우고 다음으로 넘어갑니다.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- 최종 결과 ---
    print("\n" + "="*30)
    print("모든 모델 로드 테스트 완료")
    print(f"총 {len(all_loader_functions)}개 중 {success_count}개 성공")
    if failed_models:
        print(f"실패한 모델: {failed_models}")
    print("="*30)


if __name__ == "__main__":
    test_all_models_sequentially()