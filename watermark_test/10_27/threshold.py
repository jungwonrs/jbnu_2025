import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys
from collections import defaultdict

# ✅ [수정] config.py 대신 새로운 standalone_utils.py에서 모든 것을 가져옵니다.
from analysis_utils import (
    load_test_images, load_net, coeff2img, extract, wm_metrics, spatial2coeff,
    jpeg, gblur, add_noise,
    DEVICE, WAVELET, WM_LEN, WM_SEED, WM_STRENGTH,
    JPEG_Q, GB_SIG, GN_SIGMA
)

# ---------------------------------------------------------------------------- #
#                           ✨ 실험 환경 설정 ✨                           #
# ---------------------------------------------------------------------------- #

MODEL_FILE_PATH = Path("C:\\Users\\seo\\Desktop\\watermark_experiment\\threshold_test\\inn_both.pth")
NUM_TEST_IMAGES = 10000
NUM_IMPOSTER_WATERMARKS = 10000
ROOT_DIR = Path(__file__).resolve().parent
RESULT_PLOT_FILE = ROOT_DIR / "standalone_p_au_far_frr_curve.png"
LATEX_DATA_FILE = ROOT_DIR / "standalone_p_au_latex_data.txt"

# ... (1단계: 사용자 설정 및 가중치 계산 부분은 동일) ...
p, g, sk, se = 65537, 3, 12345, 54321
pv, fv = pow(g, sk, p), pow(g, se, p)
w_ac, w_nc = pv / p, fv / p
bs = w_ac + w_nc

print("="*40)
print("사용자 설정 및 가중치 계산 완료 (1단계)")
print(f"w_ac = {w_ac:.4f}, w_nc = {w_nc:.4f}, bs = {bs:.4f}")
print("="*40)

# ... (Helper 함수 및 Main 분석 스크립트 부분은 이전과 동일) ...
def make_imposter_watermarks(n, seed_start=1000):
    imposter_wms = []
    print(f"👥 {n}개의 공격자 워터마크를 생성합니다...")
    for i in tqdm(range(n)):
        rng = np.random.RandomState(seed_start + i)
        bits = rng.randint(0, 2, WM_LEN, dtype=np.uint8)
        imposter_wms.append(bits)
    return np.array(imposter_wms)

def calculate_p_au(ac, nc):
    numerator = (w_ac * ac) + (w_nc * nc)
    if bs == 0: return 0.0
    return numerator / bs

if __name__ == "__main__":
    if not MODEL_FILE_PATH.exists(): sys.exit(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_FILE_PATH}")
    print(f"✅ 모델 파일을 불러옵니다: {MODEL_FILE_PATH}")
    netF = load_net(MODEL_FILE_PATH)
    print("\n🔬 논문 기반 `p_au` 점수를 사용한 FAR/FRR 분석을 시작합니다.")

    attack_scenarios = [('clean', lambda img: img)]
    attack_scenarios.extend([(f'jpeg{q}', lambda img, q=q: jpeg(img, q)) for q in JPEG_Q])
    attack_scenarios.extend([(f'blur{sg}', lambda img, sg=sg: (gblur(img.astype(np.float32)/255., sg)*255).round().astype(np.uint8)) for sg in GB_SIG])
    attack_scenarios.extend([(f'noise{int(sg*100)}', lambda img, sg=sg: add_noise(img, sg)) for sg in GN_SIGMA])

    # ✅ [수정] NUM_TEST_IMAGES 값을 인자로 전달
    test_images = load_test_images(NUM_TEST_IMAGES)
    
    rng_genuine = np.random.RandomState(WM_SEED)
    genuine_wm_bits = rng_genuine.randint(0, 2, WM_LEN, dtype=np.uint8)
    imposter_wms = make_imposter_watermarks(NUM_IMPOSTER_WATERMARKS)
    
    genuine_p_au_scores, imposter_p_au_scores = [], []

    print(f"🚀 {len(test_images)}개 이미지에 대해 {len(attack_scenarios)}가지 공격 시나리오를 테스트합니다...")
    
    total_tasks = len(test_images) * len(attack_scenarios)
    with tqdm(total=total_tasks, desc="전체 공격 시나리오 처리 중") as pbar:
        for img_name, tensors in test_images.items():
            try:
                # ------------------------------------------------------------------ #
                # ✅ [핵심 수정] 
                # 1. 모델이 2채널이라고 가정
                is_two_channel_model = True 
                
                # 2. 원본 gray 이미지만 가져오기
                gray_img = tensors['gray']
                
                # 3. tensors['FULL']을 무시하고, 올바른 mode로 input_coeff를 *직접* 생성
                mode = "2" if is_two_channel_model else "4"
                input_coeff = spatial2coeff(gray_img, mode) 
                # ------------------------------------------------------------------ #

                with torch.no_grad():
                    # 이제 2채널 input_coeff가 2채널 netF로 올바르게 전달됨
                    z_base, _ = netF(input_coeff)
                    
                    mid = WM_LEN // 2
                    bitsA, bitsB = genuine_wm_bits[:mid], genuine_wm_bits[mid:]
                    mapA, mapB = np.tile(bitsA, (128*128 // len(bitsA)) + 1)[:128*128].reshape(128, 128), np.tile(bitsB, (128*128 // len(bitsB)) + 1)[:128*128].reshape(128, 128)
                    wm_tA, wm_tB = torch.from_numpy((mapA*2-1) * WM_STRENGTH).float().to(DEVICE), torch.from_numpy((mapB*2-1) * WM_STRENGTH).float().to(DEVICE)
                    z_emb = z_base.clone()

                    # ✅ [수정] 2채널 모델이므로 채널 0과 1에 삽입
                    if is_two_channel_model:
                        z_emb[:, 0], z_emb[:, 1] = z_emb[:, 0] + wm_tA, z_emb[:, 1] + wm_tB
                    else:
                        z_emb[:, 1], z_emb[:, 2] = z_emb[:, 1] + wm_tA, z_emb[:, 2] + wm_tB
                        
                    stego_output = netF(z_emb, rev=True)
                    stego_coeff = stego_output[0] if isinstance(stego_output, tuple) else stego_output
                    recF = coeff2img(stego_coeff, gray_img)
                    stego_u8 = (np.clip(recF, 0, 1) * 255).round().astype(np.uint8)

                for attack_name, attack_func in attack_scenarios:
                    try:
                        attacked_u8 = attack_func(stego_u8)
                        
                        # 이 mode는 `spatial2coeff`와 `extract`에서 올바르게 사용됨
                        mode = "2" if is_two_channel_model else "4"
                        attacked_f = attacked_u8.astype(np.float32) / 255.0
                        attacked_coeff = spatial2coeff(attacked_f, mode)
                        
                        with torch.no_grad():
                            extracted_bits = extract(attacked_coeff, netF, is_two_channel_model, z_base)

                        acc_gen, _, nc_gen = wm_metrics(extracted_bits, genuine_wm_bits)
                        genuine_p_au_scores.append(calculate_p_au(acc_gen, nc_gen))

                        imposter_wm_sample = imposter_wms[np.random.randint(len(imposter_wms))]
                        acc_imp, _, nc_imp = wm_metrics(extracted_bits, imposter_wm_sample)
                        imposter_p_au_scores.append(calculate_p_au(acc_imp, nc_imp))

                    except Exception as e_attack:
                        pbar.write(f"\n⚠️ {img_name}의 '{attack_name}' 공격 처리 중 오류: {e_attack}")
                    pbar.update(1) # 공격 1회 완료
                    
            except Exception as e_image:
                pbar.write(f"\n⚠️ {img_name} 이미지 처리 중 심각한 오류: {e_image}")
                # 해당 이미지의 남은 공격 수만큼 pbar를 강제로 업데이트
                remaining_attacks = len(attack_scenarios) - (pbar.n % len(attack_scenarios))
                if remaining_attacks < len(attack_scenarios):
                     pbar.update(remaining_attacks)
                else:
                    # (만약 첫 번째 공격에서 실패했다면)
                     pbar.update(len(attack_scenarios))


    # ... (이하 결과 분석 및 출력, 그래프 생성 코드는 모두 동일) ...
    genuine_p_au_scores, imposter_p_au_scores = np.array(genuine_p_au_scores), np.array(imposter_p_au_scores)
    
    if len(genuine_p_au_scores) == 0: sys.exit("\n❌ 계산된 점수가 없습니다. (모든 이미지 처리 실패)")

    print("\n📊 `p_au` 점수 기반 최종 FAR/FRR 분석을 수행합니다.")
    min_score, max_score = min(genuine_p_au_scores.min(), imposter_p_au_scores.min()), max(genuine_p_au_scores.max(), imposter_p_au_scores.max())
    thresholds = np.linspace(min_score, max_score, 401)
    far_rates, frr_rates = [], []
    
    for t in thresholds:
        far = np.sum(imposter_p_au_scores > t) / len(imposter_p_au_scores)
        frr = np.sum(genuine_p_au_scores <= t) / len(genuine_p_au_scores)
        far_rates.append(far); frr_rates.append(frr)
        
    eer_idx = np.argmin(np.abs(np.array(far_rates) - np.array(frr_rates)))
    optimal_tau_au, eer_value = thresholds[eer_idx], (far_rates[eer_idx] + frr_rates[eer_idx]) / 2
    
    print("\n" + "="*55 + f"\n           🔬 논문 기반 최종 분석 결과 (`τ_au` 발견) 🔬\n" + "="*55)
    print(f"  - 최적 임계값 (τ_au) : {optimal_tau_au:.4f}\n  - 종합 동일 오류율 (EER) : {eer_value * 100:.4f} %")
    print("="*55)
    
    print(f"\n💾 LaTex용 Raw 데이터를 '{LATEX_DATA_FILE}' 파일에 저장합니다...")
    with open(LATEX_DATA_FILE, 'w') as f:
        f.write("Threshold_p_au\tFAR\tFRR\n")
        for i in range(len(thresholds)): f.write(f"{thresholds[i]:.6f}\t{far_rates[i]:.6f}\t{frr_rates[i]:.6f}\n")
    print("✅ 데이터 저장이 완료되었습니다.")
    
    plt.figure(figsize=(10, 6)); plt.plot(thresholds, np.array(far_rates) * 100, label='FAR (False Acceptance Rate)', color='red'); plt.plot(thresholds, np.array(frr_rates) * 100, label='FRR (False Rejection Rate)', color='blue')
    plt.scatter(optimal_tau_au, eer_value * 100, color='green', zorder=5, s=100, label=f'EER Point (τ_au)'); plt.axvline(x=optimal_tau_au, color='gray', linestyle='--', label=f'Optimal τ_au = {optimal_tau_au:.3f}')
    plt.title('FAR/FRR Curve for p_au Score'); plt.xlabel('Threshold (p_au score)'); plt.ylabel('Error Rate (%)'); plt.legend(); plt.grid(True, alpha=0.5); plt.ylim(bottom=0); plt.savefig(RESULT_PLOT_FILE)
    
    print(f"📈 결과 그래프를 '{RESULT_PLOT_FILE}' 파일로 저장했습니다.")