import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys
from collections import defaultdict

# standalone_utils.py에서 모든 것을 가져옵니다.
from analysis_utils import (
    load_test_images, load_net, coeff2img, extract, wm_metrics, spatial2coeff,
    jpeg, gblur, add_noise,
    DEVICE, WAVELET, WM_LEN, WM_SEED, WM_STRENGTH, LOGIT_COEFF,
    JPEG_Q, GB_SIG, GN_SIGMA
)

# ---------------------------------------------------------------------------- #
#                               ✨ 실험 환경 설정 ✨                           #
# ---------------------------------------------------------------------------- #

# ✅ 모델 경로 확인 (inn_both.pth 사용)
MODEL_FILE_PATH = Path("C:\\Users\\seo\\Desktop\\watermark_experiment\\threshold_test\\inn_both.pth")

NUM_TEST_IMAGES = 10000
NUM_IMPOSTER_WATERMARKS = 100000

# 가중치 후보 자동 생성 (0.0부터 1.0까지 0.01 간격)
WEIGHT_CANDIDATES = [(round(delta, 2), round(1.0 - delta, 2))
                     for delta in np.arange(0.0, 1.01, 0.01)]

ROOT_DIR = Path(__file__).resolve().parent
RESULT_PLOT_FILE = ROOT_DIR / "optimal_both_sp_far_frr_curve.png"
LATEX_DATA_FILE = ROOT_DIR / "optimal_both_sp_latex_data.txt"
ALL_EER_RESULTS_FILE = ROOT_DIR / "all_weight_eer_results.txt" # 모든 결과 저장 파일

# ---------------------------------------------------------------------------- #
#                         HELPER FUNCTIONS (내부 함수)                         #
# ---------------------------------------------------------------------------- #

def make_imposter_watermarks(n, seed_start=1000):
    imposter_wms = []
    # print(f"👥 {n}개의 공격자 워터마크를 생성합니다...") # tqdm 사용 시 중복 출력 방지
    for i in tqdm(range(n), desc=f"👥 {n}개 공격자 워터마크 생성 중"):
        rng = np.random.RandomState(seed_start + i)
        bits = rng.randint(0, 2, WM_LEN, dtype=np.uint8)
        imposter_wms.append(bits)
    return np.array(imposter_wms)

def calculate_s_p(ac, nc, delta, epsilon):
    return (delta * ac) + (epsilon * nc)

def analyze_far_frr(genuine_scores, imposter_scores):
    """
    ✅ [수정] EER 지점에서의 FAR, FRR 값을 추가로 반환합니다.
    반환값: (optimal_threshold, eer_value, far_at_eer, frr_at_eer, thresholds, far_rates, frr_rates)
    """
    if len(genuine_scores) == 0:
        # ✅ [수정] 반환값 개수 맞추기
        return None, float('inf'), None, None, None, None, None
        
    min_score = min(genuine_scores.min(), imposter_scores.min() if len(imposter_scores)>0 else genuine_scores.min())
    max_score = max(genuine_scores.max(), imposter_scores.max() if len(imposter_scores)>0 else genuine_scores.max())
    
    if abs(max_score - min_score) < 1e-6:
        min_score -= 0.05
        max_score += 0.05
        
    thresholds = np.linspace(min_score, max_score, 401)
    far_rates, frr_rates = [], []
    
    for t in thresholds:
        far = np.sum(imposter_scores > t) / len(imposter_scores) if len(imposter_scores) > 0 else 0
        frr = np.sum(genuine_scores <= t) / len(genuine_scores) if len(genuine_scores) > 0 else 0
        far_rates.append(far); frr_rates.append(frr)
        
    far_rates, frr_rates = np.array(far_rates), np.array(frr_rates)
    
    eer_idx = np.argmin(np.abs(far_rates - frr_rates))
    optimal_threshold = thresholds[eer_idx]
    
    # ✅ [수정] EER 지점에서의 FAR, FRR 값 계산
    far_at_eer = far_rates[eer_idx]
    frr_at_eer = frr_rates[eer_idx]
    eer_value = (far_at_eer + frr_at_eer) / 2
    
    # ✅ [수정] far_at_eer, frr_at_eer 추가 반환
    return optimal_threshold, eer_value, far_at_eer, frr_at_eer, thresholds, far_rates, frr_rates

# ---------------------------------------------------------------------------- #
#                       MAIN ANALYSIS SCRIPT (분석 스크립트)                      #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    if not MODEL_FILE_PATH.exists(): sys.exit(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_FILE_PATH}")
    print(f"✅ 모델 파일을 불러옵니다: {MODEL_FILE_PATH}")
    netF = load_net(MODEL_FILE_PATH)
    print(f"\n🔬 최적의 가중치(δ, ε)와 임계값(τ_p) 자동 탐색 (0.01 간격)을 시작합니다...")

    # ... (공격 시나리오, 데이터 로딩, ac/nc 계산 부분은 동일) ...
    attack_scenarios = [('clean', lambda img: img)]
    attack_scenarios.extend([(f'jpeg{q}', lambda img, q=q: jpeg(img, q)) for q in JPEG_Q])
    attack_scenarios.extend([(f'blur{sg}', lambda img, sg=sg: (gblur(img.astype(np.float32)/255., sg)*255).round().astype(np.uint8)) for sg in GB_SIG])
    attack_scenarios.extend([(f'noise{int(sg*100)}', lambda img, sg=sg: add_noise(img, sg)) for sg in GN_SIGMA])
    test_images = load_test_images(NUM_TEST_IMAGES)
    rng_genuine = np.random.RandomState(WM_SEED)
    genuine_wm_bits = rng_genuine.randint(0, 2, WM_LEN, dtype=np.uint8)
    imposter_wms = make_imposter_watermarks(NUM_IMPOSTER_WATERMARKS)
    genuine_metrics, imposter_metrics = [], []
    print(f"🚀 {len(test_images)}개 이미지 * {len(attack_scenarios)}가지 공격에 대한 ac, nc 값을 먼저 계산합니다...")
    total_tasks = len(test_images) * len(attack_scenarios)
    with tqdm(total=total_tasks, desc="ac, nc 계산 중") as pbar:
        for img_name, tensors in test_images.items():
            # ... (내부 이미지 처리 및 ac/nc 계산 루프는 동일) ...
            try:
                is_two_channel_model = True; gray_img = tensors['gray']; input_coeff = tensors['LH_HL']
                with torch.no_grad():
                    z_base, _ = netF(input_coeff)
                    mid = WM_LEN // 2; bitsA, bitsB = genuine_wm_bits[:mid], genuine_wm_bits[mid:]
                    mapA, mapB = np.tile(bitsA, (128*128 // len(bitsA)) + 1)[:128*128].reshape(128, 128), np.tile(bitsB, (128*128 // len(bitsB)) + 1)[:128*128].reshape(128, 128)
                    wm_tA, wm_tB = torch.from_numpy((mapA*2-1) * WM_STRENGTH).float().to(DEVICE), torch.from_numpy((mapB*2-1) * WM_STRENGTH).float().to(DEVICE)
                    z_emb = z_base.clone(); z_emb[:, 0], z_emb[:, 1] = z_emb[:, 0] + wm_tA, z_emb[:, 1] + wm_tB
                    stego_output = netF(z_emb, rev=True); stego_coeff = stego_output[0] if isinstance(stego_output, tuple) else stego_output
                    recF = coeff2img(stego_coeff, gray_img); stego_u8 = (np.clip(recF, 0, 1) * 255).round().astype(np.uint8)
                for attack_name, attack_func in attack_scenarios:
                    try:
                        attacked_u8 = attack_func(stego_u8); mode = "2"
                        attacked_f = attacked_u8.astype(np.float32) / 255.0; attacked_coeff = spatial2coeff(attacked_f, mode)
                        with torch.no_grad(): extracted_bits = extract(attacked_coeff, netF, is_two_channel_model, z_base)
                        acc_gen, _, nc_gen = wm_metrics(extracted_bits, genuine_wm_bits); genuine_metrics.append((acc_gen, nc_gen))
                        imposter_wm_sample = imposter_wms[np.random.randint(len(imposter_wms))]
                        acc_imp, _, nc_imp = wm_metrics(extracted_bits, imposter_wm_sample); imposter_metrics.append((acc_imp, nc_imp))
                    except Exception as e_attack: pbar.write(f"\n⚠️ {img_name} '{attack_name}' 오류: {e_attack}")
                    pbar.update(1)
            except Exception as e_image:
                pbar.write(f"\n⚠️ {img_name} 처리 오류: {e_image}")
                remaining_attacks = len(attack_scenarios) - (pbar.n % len(attack_scenarios)); pbar.update(remaining_attacks)

    if not genuine_metrics: sys.exit("\n❌ 계산된 (ac, nc) 값이 없습니다.")

    best_eer = float('inf'); best_weights = (0, 0); best_tau_p = 0
    best_analysis_results = None
    all_results_data = {} # ✅ [수정] (tau_p, eer, far, frr) 저장용 딕셔너리

    print(f"\n📊 {len(WEIGHT_CANDIDATES)}개의 가중치 조합(0.01 간격)에 대해 최적의 τ_p 와 EER을 탐색합니다...")
    for delta, epsilon in tqdm(WEIGHT_CANDIDATES, desc="가중치 조합 탐색 중"):
        genuine_sp_scores = np.array([calculate_s_p(ac, nc, delta, epsilon) for ac, nc in genuine_metrics])
        imposter_sp_scores = np.array([calculate_s_p(ac, nc, delta, epsilon) for ac, nc in imposter_metrics])
        
        # ✅ [수정] far_at_eer, frr_at_eer 값도 함께 받음
        tau_p, eer, far_at_eer, frr_at_eer, thresholds, far_rates, frr_rates = analyze_far_frr(genuine_sp_scores, imposter_sp_scores)

        # ✅ [수정] (tau_p, eer, far_at_eer, frr_at_eer) 4개 값 저장
        if tau_p is not None: # analyze_far_frr이 정상 실행되었을 경우만 저장
            all_results_data[(delta, epsilon)] = (tau_p, eer, far_at_eer, frr_at_eer)

            if eer < best_eer:
                best_eer, best_weights, best_tau_p = eer, (delta, epsilon), tau_p
                best_analysis_results = (thresholds, far_rates, frr_rates)
            # EER이 동일할 경우, delta가 0.5에 가까운 것을 우선 선택
            elif abs(eer - best_eer) < 1e-6 and abs(delta - 0.5) < abs(best_weights[0] - 0.5):
                best_weights, best_tau_p = (delta, epsilon), tau_p
                best_analysis_results = (thresholds, far_rates, frr_rates)

    # ... (최종 결과 출력 부분은 동일) ...
    print("\n" + "="*60 + "\n            ✨ 최종 최적 가중치 및 임계값 발견 결과 ✨\n" + "="*60)
    print(f"  - 최적 가중치 (δ, ε)  : ({best_weights[0]:.2f}, {best_weights[1]:.2f})")
    print(f"  - 최적 임계값 (τ_p)    : {best_tau_p:.4f}")
    print(f"  - 최소 동일 오류율 (EER) : {best_eer * 100:.4f} %")
    print("="*60)

    # ✅ [수정] 모든 가중치 조합별 EER, tau_p, FAR, FRR 결과를 파일에 저장
    print(f"\n💾 모든 가중치 조합별 EER, τ_p, FAR, FRR 결과를 '{ALL_EER_RESULTS_FILE}' 파일에 저장합니다...")
    try:
        with open(ALL_EER_RESULTS_FILE, 'w') as f:
            # ✅ [수정] 헤더에 FAR, FRR 열 추가
            f.write("Delta (δ)\tEpsilon (ε)\tEER (%)\tOptimal Tau (τ_p)\tFAR_at_EER (%)\tFRR_at_EER (%)\n")
            
            # all_results_data 딕셔너리를 키(가중치 튜플) 기준으로 정렬하여 파일에 쓰기
            # ✅ [수정] far_val, frr_val 값도 함께 읽어와서 저장
            for weights, (tau_p_val, eer_val, far_val, frr_val) in sorted(all_results_data.items()):
                f.write(f"{weights[0]:.2f}\t{weights[1]:.2f}\t{eer_val*100:.4f}\t{tau_p_val:.4f}\t{far_val*100:.4f}\t{frr_val*100:.4f}\n")
        print("✅ 모든 결과 저장이 완료되었습니다.")
    except Exception as e:
        print(f"❌ 모든 결과 저장 중 오류 발생: {e}")

    # ... (최적 조합 LaTex 데이터 저장 및 그래프 생성 코드는 동일) ...
    if best_analysis_results:
        thresholds, far_rates, frr_rates = best_analysis_results
        print(f"\n💾 최적 조합(δ={best_weights[0]:.2f}, ε={best_weights[1]:.2f})의 LaTex 데이터를 '{LATEX_DATA_FILE}' 파일에 저장합니다...")
        with open(LATEX_DATA_FILE, 'w') as f:
            f.write("Threshold_sp\tFAR\tFRR\n")
            for i in range(len(thresholds)): f.write(f"{thresholds[i]:.6f}\t{far_rates[i]:.6f}\t{frr_rates[i]:.6f}\n")
        print("✅ 데이터 저장이 완료되었습니다.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, far_rates * 100, label='FAR (False Acceptance Rate)', color='red')
        plt.plot(thresholds, frr_rates * 100, label='FRR (False Rejection Rate)', color='blue')
        plt.scatter(best_tau_p, best_eer * 100, color='green', zorder=5, s=100, label=f'EER Point ({best_eer * 100:.2f}%)')
        plt.axvline(x=best_tau_p, color='gray', linestyle='--', label=f'Optimal τ_p = {best_tau_p:.3f}')
        plt.title(f'Optimal FAR/FRR Curve for s_p Score (δ={best_weights[0]:.2f}, ε={best_weights[1]:.2f})')
        plt.xlabel('Threshold (s_p score)')
        plt.ylabel('Error Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.ylim(bottom=0)
        plt.savefig(RESULT_PLOT_FILE)
        print(f"📈 최적 조합의 결과 그래프를 '{RESULT_PLOT_FILE}' 파일로 저장했습니다.")
    else:
        print("❌ 분석 결과를 찾지 못해 그래프 및 데이터 파일을 생성할 수 없습니다.")