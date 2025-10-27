import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys
from collections import defaultdict

# analysis_utils.py 에서 함수 및 설정값 가져오기
from analysis_utils import (
    load_test_images, load_net, coeff2img, extract, wm_metrics, spatial2coeff,
    jpeg, gblur, add_noise,
    DEVICE, WAVELET, WM_LEN, WM_SEED, WM_STRENGTH, LOGIT_COEFF, # 필요한 설정값들 추가
    JPEG_Q, GB_SIG, GN_SIGMA
)

# ---------------------------------------------------------------------------- #
#                           ✨ 실험 환경 설정 ✨                               #
# ---------------------------------------------------------------------------- #

MODEL_FILE_PATH = Path("C:\\Users\\seo\\Desktop\\watermark_experiment\\threshold_test\\inn_both.pth")
NUM_TEST_IMAGES = 10000
NUM_IMPOSTER_WATERMARKS = 10000

# 논문에 따른 유사도 점수 가중치 (δ, ε) 설정 (δ + ε = 1)
DELTA_WEIGHT = 0.5  # δ (ac 가중치)
EPSILON_WEIGHT = 0.5 # ε (nc 가중치)
assert DELTA_WEIGHT + EPSILON_WEIGHT == 1.0, "Delta와 Epsilon의 합은 1이어야 합니다."

ROOT_DIR = Path(__file__).resolve().parent
RESULT_PLOT_FILE = ROOT_DIR / "both_model_sp_far_frr_curve.png" # 파일 이름 변경 (both 모델 명시)
LATEX_DATA_FILE = ROOT_DIR / "both_model_sp_latex_data.txt"    # 파일 이름 변경 (both 모델 명시)

# ---------------------------------------------------------------------------- #
#                         HELPER FUNCTIONS (내부 함수)                         #
# ---------------------------------------------------------------------------- #

def make_imposter_watermarks(n, seed_start=1000):
    imposter_wms = []
    print(f"👥 {n}개의 공격자 워터마크를 생성합니다...")
    for i in tqdm(range(n)):
        rng = np.random.RandomState(seed_start + i)
        bits = rng.randint(0, 2, WM_LEN, dtype=np.uint8)
        imposter_wms.append(bits)
    return np.array(imposter_wms)

def calculate_s_p(ac, nc):
    return (DELTA_WEIGHT * ac) + (EPSILON_WEIGHT * nc)

# ---------------------------------------------------------------------------- #
#                       MAIN ANALYSIS SCRIPT (분석 스크립트)                   #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    if not MODEL_FILE_PATH.exists(): sys.exit(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_FILE_PATH}")
    print(f"✅ 모델 파일을 불러옵니다: {MODEL_FILE_PATH}")
    netF = load_net(MODEL_FILE_PATH) # utils의 load_net 사용 (채널 자동 감지)
    print(f"\n🔬 논문 기반 `s_p` 점수 (δ={DELTA_WEIGHT}, ε={EPSILON_WEIGHT})를 사용한 FAR/FRR 분석을 시작합니다.")

    attack_scenarios = [('clean', lambda img: img)]
    attack_scenarios.extend([(f'jpeg{q}', lambda img, q=q: jpeg(img, q)) for q in JPEG_Q])
    attack_scenarios.extend([(f'blur{sg}', lambda img, sg=sg: (gblur(img.astype(np.float32)/255., sg)*255).round().astype(np.uint8)) for sg in GB_SIG])
    attack_scenarios.extend([(f'noise{int(sg*100)}', lambda img, sg=sg: add_noise(img, sg)) for sg in GN_SIGMA])

    test_images = load_test_images(NUM_TEST_IMAGES)
    rng_genuine = np.random.RandomState(WM_SEED)
    genuine_wm_bits = rng_genuine.randint(0, 2, WM_LEN, dtype=np.uint8)
    imposter_wms = make_imposter_watermarks(NUM_IMPOSTER_WATERMARKS)

    genuine_sp_scores, imposter_sp_scores = [], []
    results_by_attack = defaultdict(lambda: defaultdict(list))

    print(f"🚀 {len(test_images)}개 이미지에 대해 {len(attack_scenarios)}가지 공격 시나리오를 테스트합니다...")

    total_tasks = len(test_images) * len(attack_scenarios)
    with tqdm(total=total_tasks, desc="전체 공격 시나리오 처리 중") as pbar:
        for img_name, tensors in test_images.items():
            try:
                # ✅ [핵심 수정] inn_both 모델은 2채널이므로 is_two_channel_model = True
                is_two_channel_model = True
                gray_img = tensors['gray']
                
                # ✅ [핵심 수정] 2채널 모델에 맞는 'LH_HL' 텐서를 입력으로 사용
                input_coeff = tensors['LH_HL'] 
                
                with torch.no_grad():
                    z_base, _ = netF(input_coeff) # 2채널 입력 -> 2채널 모델
                    mid = WM_LEN // 2
                    bitsA, bitsB = genuine_wm_bits[:mid], genuine_wm_bits[mid:]
                    mapA, mapB = np.tile(bitsA, (128*128 // len(bitsA)) + 1)[:128*128].reshape(128, 128), np.tile(bitsB, (128*128 // len(bitsB)) + 1)[:128*128].reshape(128, 128)
                    wm_tA, wm_tB = torch.from_numpy((mapA*2-1) * WM_STRENGTH).float().to(DEVICE), torch.from_numpy((mapB*2-1) * WM_STRENGTH).float().to(DEVICE)
                    z_emb = z_base.clone()

                    # ✅ [핵심 수정] 2채널 모델이므로 채널 0과 1에 삽입
                    z_emb[:, 0], z_emb[:, 1] = z_emb[:, 0] + wm_tA, z_emb[:, 1] + wm_tB
                        
                    stego_output = netF(z_emb, rev=True)
                    stego_coeff = stego_output[0] if isinstance(stego_output, tuple) else stego_output
                    recF = coeff2img(stego_coeff, gray_img) # coeff2img는 2채널 입력 처리 가능
                    stego_u8 = (np.clip(recF, 0, 1) * 255).round().astype(np.uint8)

                for attack_name, attack_func in attack_scenarios:
                    try:
                        attacked_u8 = attack_func(stego_u8)
                        
                        # ✅ [핵심 수정] 2채널 모델이므로 mode="2" 사용
                        mode = "2" if is_two_channel_model else "4" 
                        attacked_f = attacked_u8.astype(np.float32) / 255.0
                        attacked_coeff = spatial2coeff(attacked_f, mode) # 2채널 계수 생성
                        
                        with torch.no_grad():
                            # ✅ [핵심 수정] extract 함수에 is_two_channel_model=True 전달
                            extracted_bits = extract(attacked_coeff, netF, is_two_channel_model, z_base)

                        acc_gen, ber_gen, nc_gen = wm_metrics(extracted_bits, genuine_wm_bits)
                        s_p_genuine = calculate_s_p(acc_gen, nc_gen)
                        genuine_sp_scores.append(s_p_genuine)

                        results_by_attack[attack_name]['acc'].append(acc_gen)
                        results_by_attack[attack_name]['ber'].append(ber_gen)
                        results_by_attack[attack_name]['nc'].append(nc_gen)

                        imposter_wm_sample = imposter_wms[np.random.randint(len(imposter_wms))]
                        acc_imp, _, nc_imp = wm_metrics(extracted_bits, imposter_wm_sample)
                        s_p_imposter = calculate_s_p(acc_imp, nc_imp)
                        imposter_sp_scores.append(s_p_imposter)

                    except Exception as e_attack:
                        pbar.write(f"\n⚠️ {img_name}의 '{attack_name}' 공격 처리 중 오류: {e_attack}")
                    pbar.update(1)
            except Exception as e_image:
                pbar.write(f"\n⚠️ {img_name} 이미지 처리 중 심각한 오류: {e_image}")
                remaining_attacks = len(attack_scenarios) - (pbar.n % len(attack_scenarios))
                pbar.update(remaining_attacks)

    # ... (이하 결과 분석 및 출력, 그래프 생성 코드는 모두 동일) ...
    print("\n" + "="*60 + "\n              📊 각 공격 시나리오별 세부 결과 (ac, nc 기반) 📊\n" + "="*60)
    header = "{:<12} | {:>10} | {:>10} | {:>10}".format("Attack", "Avg-ACC", "Avg-BER", "Avg-NC")
    print(header); print("-" * len(header))
    for name, _ in attack_scenarios:
        if name in results_by_attack and results_by_attack[name]['acc']:
            avg_acc, avg_ber, avg_nc = np.mean(results_by_attack[name]['acc']), np.mean(results_by_attack[name]['ber']), np.mean(results_by_attack[name]['nc'])
            print("{:<12} | {:10.4f} | {:10.4f} | {:10.4f}".format(name, avg_acc, avg_ber, avg_nc))
    print("="*60)
    genuine_sp_scores, imposter_sp_scores = np.array(genuine_sp_scores), np.array(imposter_sp_scores)
    if len(genuine_sp_scores) == 0: sys.exit("\n❌ 계산된 점수가 없습니다.")
    print("\n📊 `s_p` 점수 기반 최종 FAR/FRR 분석을 수행합니다.")
    min_score, max_score = min(genuine_sp_scores.min(), imposter_sp_scores.min()), max(genuine_sp_scores.max(), imposter_sp_scores.max())
    if max_score - min_score < 0.1: max_score = min_score + 0.1
    thresholds = np.linspace(min_score, max_score, 401)
    far_rates, frr_rates = [], []
    for t in thresholds:
        far = np.sum(imposter_sp_scores > t) / len(imposter_sp_scores); frr = np.sum(genuine_sp_scores <= t) / len(genuine_sp_scores)
        far_rates.append(far); frr_rates.append(frr)
    eer_idx = np.argmin(np.abs(np.array(far_rates) - np.array(frr_rates)))
    optimal_tau_p, eer_value = thresholds[eer_idx], (far_rates[eer_idx] + frr_rates[eer_idx]) / 2
    print("\n" + "="*55 + f"\n           🔬 논문 기반 최종 분석 결과 (`τ_p` 발견) 🔬\n" + "="*55)
    print(f"  - 최적 임계값 (τ_p)    : {optimal_tau_p:.4f}\n  - 종합 동일 오류율 (EER) : {eer_value * 100:.4f} %")
    print("="*55)
    print(f"\n💾 LaTex용 Raw 데이터를 '{LATEX_DATA_FILE}' 파일에 저장합니다...")
    with open(LATEX_DATA_FILE, 'w') as f:
        f.write("Threshold_sp\tFAR\tFRR\n")
        for i in range(len(thresholds)): f.write(f"{thresholds[i]:.6f}\t{far_rates[i]:.6f}\t{frr_rates[i]:.6f}\n")
    print("✅ 데이터 저장이 완료되었습니다.")
    plt.figure(figsize=(10, 6)); plt.plot(thresholds, np.array(far_rates) * 100, label='FAR (False Acceptance Rate)', color='red'); plt.plot(thresholds, np.array(frr_rates) * 100, label='FRR (False Rejection Rate)', color='blue')
    plt.scatter(optimal_tau_p, eer_value * 100, color='green', zorder=5, s=100, label=f'EER Point (τ_p)'); plt.axvline(x=optimal_tau_p, color='gray', linestyle='--', label=f'Optimal τ_p = {optimal_tau_p:.3f}')
    plt.title('FAR/FRR Curve for s_p Score'); plt.xlabel('Threshold (s_p score)'); plt.ylabel('Error Rate (%)'); plt.legend(); plt.grid(True, alpha=0.5); plt.ylim(bottom=0); plt.savefig(RESULT_PLOT_FILE)
    print(f"📈 결과 그래프를 '{RESULT_PLOT_FILE}' 파일로 저장했습니다.")