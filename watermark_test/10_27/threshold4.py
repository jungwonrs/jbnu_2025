import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys
import os

# ---------------------------------------------------------------------------- #
#                ✅ 실제 analysis_utils 모듈 임포트                             #
# ---------------------------------------------------------------------------- #
try:
    from analysis_utils import (
        load_test_images, load_net, coeff2img, extract, wm_metrics, spatial2coeff,
        jpeg, gblur, add_noise,
        DEVICE, WAVELET, WM_LEN, WM_SEED, WM_STRENGTH, LOGIT_COEFF,
        JPEG_Q, GB_SIG, GN_SIGMA
    )
except ImportError:
    sys.exit("❌ Error: analysis_utils.py 파일을 찾을 수 없습니다. 같은 폴더에 넣어주세요.")


# ---------------------------------------------------------------------------- #
#                        ✨ 실험 환경 설정 ✨                                  #
# ---------------------------------------------------------------------------- #

# ✅ 모델 경로를 실제 환경에 맞게 변경하세요. 
MODEL_FILE_PATH = Path("C:\\Users\\seo\\Desktop\\watermark_experiment\\threshold_test\\inn_both.pth") 

NUM_TEST_IMAGES = 100 # 테스트 이미지 수
NUM_IMPOSTER_WATERMARKS = 1000 # Imposter 워터마크 수

# 가중치 후보 자동 생성 (0.0부터 1.0까지 0.01 간격)
WEIGHT_CANDIDATES = [(round(delta, 2), round(1.0 - delta, 2))
                     for delta in np.arange(0.0, 1.01, 0.01)]

ROOT_DIR = Path(__file__).resolve().parent if '__file__' in locals() else Path(".")

# ---------------------------------------------------------------------------- #
#                        HELPER FUNCTIONS (내부 함수)                        #
# ---------------------------------------------------------------------------- #

def make_imposter_watermarks(n, seed_start=1000):
    imposter_wms = []
    rng_fixed = np.random.RandomState(seed_start)
    genuine_wm_fixed = np.random.RandomState(WM_SEED).randint(0, 2, WM_LEN, dtype=np.uint8)

    for i in tqdm(range(n), desc=f"👥 {n}개 공격자 워터마크 생성 중"):
        bits = rng_fixed.randint(0, 2, WM_LEN, dtype=np.uint8)
        
        if np.array_equal(bits, genuine_wm_fixed):
            bits[0] = 1 - bits[0]
            
        imposter_wms.append(bits)
    return np.array(imposter_wms)

def calculate_s_p(ac, nc, delta, epsilon, mode='LINEAR'):
    """
    ac와 nc를 사용하여 EER 개선을 위한 유사도 점수(s_p)를 계산합니다.
    """
    
    if mode == 'L2_NORM':
        # 1. L2-Norm 기반: s'_p = sqrt(delta * ac^2 + epsilon * nc^2)
        ac_sq = ac**2
        nc_sq = nc**2
        weighted_sum_sq = (delta * ac_sq) + (epsilon * nc_sq)
        return np.sqrt(np.clip(weighted_sum_sq, 0, 1))

    elif mode == 'HARMONIC':
        # 2. 조화 평균 결합: s''_p = delta * (2 * ac * nc / (ac + nc)) + epsilon * max(ac, nc)
        harmonic_mean = (2.0 * ac * nc) / (ac + nc + 1e-9)
        max_val = np.maximum(ac, nc)
        return (delta * harmonic_mean) + (epsilon * max_val)

    elif mode == 'LOG_COMP':
        # 3. 로그 보정 공식: s_p = delta * ac + epsilon * ac * log10(nc + 0.01)
        log_term = np.log10(np.clip(nc, 1e-9, 1.0) + 0.01)
        score = (delta * ac) + (epsilon * ac * log_term)
        return score 
    
    elif mode == 'DIFFERENCE':
        # 4. 차이 강조 공식: s_p = delta * ac + epsilon * (ac - nc)
        score = (delta * ac) + (epsilon * (ac - nc))
        return score
    
    # ✅ [새로운 공식 추가] 5. NC 스케일링 공식
    elif mode == 'NC_SCALED':
        # s_p = delta * ac + epsilon * (nc * 100)
        score = (delta * ac) + (epsilon * nc * 100.0)
        # 점수 범위가 매우 넓어질 수 있으므로 클리핑하지 않습니다.
        return score

    else:
        # Baseline (기존 선형 공식)
        return (delta * ac) + (epsilon * nc)

def analyze_far_frr(genuine_scores, imposter_scores):
    if len(genuine_scores) == 0: return None, float('inf'), None, None, None, None, None
    all_scores = np.concatenate([genuine_scores, imposter_scores])
    min_score, max_score = all_scores.min(), all_scores.max()
    
    score_range = max_score - min_score
    if score_range < 1e-6:
        min_score -= 0.05
        max_score += 0.05
    else:
        buffer = score_range * 0.05 
        min_score -= buffer
        max_score += buffer
        
    # 점수 범위가 100배가 될 수 있으므로, threshold 개수를 늘려 정밀도를 높입니다.
    thresholds = np.linspace(min_score, max_score, 1001) 
    far_rates, frr_rates = [], []
    
    n_imposters = len(imposter_scores)
    n_genuine = len(genuine_scores)
    
    for t in thresholds:
        far = np.sum(imposter_scores > t) / n_imposters if n_imposters > 0 else 0
        frr = np.sum(genuine_scores <= t) / n_genuine if n_genuine > 0 else 0
        far_rates.append(far); frr_rates.append(frr)
        
    far_rates, frr_rates = np.array(far_rates), np.array(frr_rates)
    eer_diff = np.abs(far_rates - frr_rates)
    eer_idx = np.argmin(eer_diff)
    
    optimal_threshold = thresholds[eer_idx]
    eer_value = (far_rates[eer_idx] + frr_rates[eer_idx]) / 2
    
    far_at_eer = far_rates[eer_idx]
    frr_at_eer = frr_rates[eer_idx]
    
    return optimal_threshold, eer_value, far_at_eer, frr_at_eer, thresholds, far_rates, frr_rates

def run_analysis_for_mode(metrics_data, mode_name, weight_candidates):
    """주어진 ac/nc 데이터에 대해 특정 모드로 EER 분석을 실행하고 결과를 저장합니다."""
    genuine_metrics = metrics_data['genuine']
    imposter_metrics = metrics_data['imposter']
    
    best_eer = float('inf'); best_weights = (0, 0); best_tau_p = 0
    best_analysis_results = None
    all_results_data = {}

    print(f"\n📊 [{mode_name} MODE] 최적의 가중치(δ, ε)와 임계값(τ_p) 자동 탐색 중...")
    for delta, epsilon in tqdm(weight_candidates, desc="가중치 조합 탐색 중"):
        genuine_sp_scores = np.array([calculate_s_p(ac, nc, delta, epsilon, mode=mode_name) for ac, nc in genuine_metrics])
        imposter_sp_scores = np.array([calculate_s_p(ac, nc, delta, epsilon, mode=mode_name) for ac, nc in imposter_metrics])
        
        results = analyze_far_frr(genuine_sp_scores, imposter_sp_scores)
        if results is None: continue
        
        tau_p, eer_val, far_at_eer, frr_at_eer, thresholds, far_rates, frr_rates = results
        
        if tau_p is not None:
            all_results_data[(delta, epsilon)] = (tau_p, eer_val, far_at_eer, frr_at_eer)
            
            # Tie-breaker logic: Prefer lowest EER, then closest delta to 0.5
            if eer_val < best_eer:
                best_eer, best_weights, best_tau_p = eer_val, (delta, epsilon), tau_p
                best_analysis_results = (thresholds, far_rates, frr_rates)
            elif abs(eer_val - best_eer) < 1e-6: 
                if abs(delta - 0.5) < abs(best_weights[0] - 0.5):
                    best_weights, best_tau_p = (delta, epsilon), tau_p
                    best_analysis_results = (thresholds, far_rates, frr_rates)
                    
    # 모든 가중치 결과를 파일로 저장
    ALL_EER_RESULTS_FILE = ROOT_DIR / f"all_weight_eer_results_{mode_name.lower()}.txt"
    try:
        with open(ALL_EER_RESULTS_FILE, 'w') as f:
            f.write("Delta (\\delta)\tEpsilon (\\epsilon)\tEER (%)\tOptimal Tau (\\tau_p)\tFAR_at_EER (%)\tFRR_at_EER (%)\n")
            
            for weights, (tau_p_val, eer_val, far_val, frr_val) in sorted(all_results_data.items()):
                f.write(f"{weights[0]:.2f}\t{weights[1]:.2f}\t{eer_val*100:.4f}\t{tau_p_val:.4f}\t{far_val*100:.4f}\t{frr_val*100:.4f}\n")
        print(f"✅ [{mode_name} MODE] 모든 가중치별 EER 결과 저장이 완료되었습니다: {ALL_EER_RESULTS_FILE.name}")
    except Exception as e:
        print(f"❌ [{mode_name} MODE] 결과 저장 중 오류 발생: {e}")
        
    return {
        'mode': mode_name,
        'best_eer': best_eer,
        'best_weights': best_weights,
        'best_tau_p': best_tau_p,
        'best_analysis_results': best_analysis_results
    }


# ---------------------------------------------------------------------------- #
#                      MAIN ANALYSIS SCRIPT (분석 스크립트)                     #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    if not MODEL_FILE_PATH.exists(): sys.exit(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_FILE_PATH}")
    print(f"✅ 모델 파일을 불러옵니다: {MODEL_FILE_PATH}")
    netF = load_net(MODEL_FILE_PATH)

    # 1. AC/NC 데이터 사전 계산 
    print(f"\n🔬 AC/NC 데이터 사전 계산을 시작합니다...")
    
    attack_scenarios = [('clean', lambda img: img)]
    attack_scenarios.extend([(f'jpeg{q}', lambda img, q=q: jpeg(img, q)) for q in JPEG_Q])
    attack_scenarios.extend([(f'blur{sg}', lambda img, sg=sg: (gblur(img.astype(np.float32)/255., sg)*255).round().astype(np.uint8)) for sg in GB_SIG])
    attack_scenarios.extend([(f'noise{int(sg*100)}', lambda img, sg=sg: add_noise(img, sg)) for sg in GN_SIGMA])
    
    try:
        test_images = load_test_images(NUM_TEST_IMAGES)
    except Exception as e:
        sys.exit(f"❌ 테스트 이미지 로딩 오류: {e}")

    rng_genuine = np.random.RandomState(WM_SEED)
    genuine_wm_bits = rng_genuine.randint(0, 2, WM_LEN, dtype=np.uint8)
    imposter_wms = make_imposter_watermarks(NUM_IMPOSTER_WATERMARKS)
    genuine_metrics, imposter_metrics = [], []
    
    total_tasks = len(test_images) * len(attack_scenarios)
    
    print(f"🚀 {len(test_images)}개 이미지 * {len(attack_scenarios)}가지 공격에 대한 ac, nc 값 계산 중 (총 {total_tasks} tasks)...")
    
    # -----------------------------------------------------------------------
    #               실제 AC/NC 계산 루프 (analysis_utils 사용)                        
    # -----------------------------------------------------------------------
    with tqdm(total=total_tasks, desc="AC, NC 계산 중") as pbar:
        for img_name, tensors in test_images.items():
            try:
                is_two_channel_model = True 
                gray_img = tensors['gray']
                input_coeff = tensors['LH_HL'] 
                
                with torch.no_grad():
                    # 1. 워터마크 삽입 과정
                    z_base, _ = netF(input_coeff)
                    
                    mid = WM_LEN // 2
                    bitsA, bitsB = genuine_wm_bits[:mid], genuine_wm_bits[mid:]
                    mapA = np.tile(bitsA, (128*128 // len(bitsA)) + 1)[:128*128].reshape(128, 128)
                    mapB = np.tile(bitsB, (128*128 // len(bitsB)) + 1)[:128*128].reshape(128, 128)
                    wm_tA = torch.from_numpy((mapA*2-1) * WM_STRENGTH).float().to(DEVICE)
                    wm_tB = torch.from_numpy((mapB*2-1) * WM_STRENGTH).float().to(DEVICE)
                    
                    z_emb = z_base.clone()
                    z_emb[:, 0] = z_emb[:, 0] + wm_tA 
                    z_emb[:, 1] = z_emb[:, 1] + wm_tB 
                    
                    stego_output = netF(z_emb, rev=True)
                    stego_coeff = stego_output[0] if isinstance(stego_output, tuple) else stego_output
                    recF = coeff2img(stego_coeff, gray_img)
                    stego_u8 = (np.clip(recF, 0, 1) * 255).round().astype(np.uint8)
                    
                # 2. 공격 및 추출 과정
                for attack_name, attack_func in attack_scenarios:
                    try:
                        attacked_u8 = attack_func(stego_u8)
                        mode = "2" # LH-HL 사용 모델이므로 DWT 2채널 모드 사용
                        attacked_f = attacked_u8.astype(np.float32) / 255.0
                        attacked_coeff = spatial2coeff(attacked_f, mode)
                        
                        with torch.no_grad(): 
                            extracted_bits = extract(attacked_coeff, netF, is_two_channel_model, z_base)
                        
                        # Genuine Score 계산
                        acc_gen, _, nc_gen = wm_metrics(extracted_bits, genuine_wm_bits)
                        genuine_metrics.append((acc_gen, nc_gen))
                        
                        # Imposter Score 계산
                        imposter_wm_sample = imposter_wms[np.random.randint(len(imposter_wms))]
                        acc_imp, _, nc_imp = wm_metrics(extracted_bits, imposter_wm_sample)
                        imposter_metrics.append((acc_imp, nc_imp))
                        
                    except Exception as e_attack: 
                        pbar.write(f"\n⚠️ {img_name} '{attack_name}' 오류: {e_attack}")
                    pbar.update(1)
                    
            except Exception as e_image:
                pbar.write(f"\n⚠️ {img_name} 처리 오류: {e_image}")
                remaining_attacks = len(attack_scenarios) - (pbar.n % len(attack_scenarios))
                pbar.update(remaining_attacks)

    if not genuine_metrics: sys.exit("\n❌ 계산된 (ac, nc) 값이 없습니다. 이미지 경로 또는 모델을 확인하세요.")
    metrics_data = {'genuine': genuine_metrics, 'imposter': imposter_metrics}

    # 2. 다섯 가지 공식에 대한 EER 분석 실행
    experiment_modes = ['L2_NORM', 'HARMONIC', 'LOG_COMP', 'DIFFERENCE', 'NC_SCALED'] # ✅ 'NC_SCALED' 추가
    all_results = []
    
    for mode in experiment_modes:
        result = run_analysis_for_mode(metrics_data, mode, WEIGHT_CANDIDATES)
        all_results.append(result)

    # 3. 결과 출력 및 비교
    print("\n" + "="*80)
    print("                      ✨ 최종 EER 개선 실험 결과 비교 ✨")
    print("="*80)
    
    best_overall_eer = float('inf')
    best_overall_result = None

    for result in all_results:
        eer_percent = result['best_eer'] * 100
        delta = result['best_weights'][0]
        epsilon = result['best_weights'][1]
        
        print(f"| [MODE: {result['mode']:<11}] | 최적 EER: {eer_percent:.4f}% | 최적 (δ, ε): ({delta:.2f}, {epsilon:.2f}) | τ_p: {result['best_tau_p']:.4f} |")
        
        if result['best_eer'] < best_overall_eer:
            best_overall_eer = result['best_eer']
            best_overall_result = result

    print("="*80)
    
    # 4. 가장 좋은 결과 저장 (LaTeX 데이터 및 그래프)
    final_result = best_overall_result
    
    if final_result:
        mode_tag = final_result['mode'].lower()
        RESULT_PLOT_FILE = ROOT_DIR / f"optimal_sp_far_frr_curve_{mode_tag}.png"
        LATEX_DATA_FILE = ROOT_DIR / f"optimal_sp_latex_data_{mode_tag}.txt"
        
        thresholds, far_rates, frr_rates = final_result['best_analysis_results']
        best_tau_p = final_result['best_tau_p']
        best_eer = final_result['best_eer']
        best_weights = final_result['best_weights']
        
        print(f"\n💾 최적 공식 ({final_result['mode']})에 대한 최종 데이터를 저장합니다.")
        
        # LaTeX 데이터 파일 저장
        with open(LATEX_DATA_FILE, 'w') as f:
            f.write("Threshold_sp\tFAR\tFRR\n")
            for i in range(len(thresholds)): 
                f.write(f"{thresholds[i]:.6f}\t{far_rates[i]:.6f}\t{frr_rates[i]:.6f}\n")
        print(f"✅ LaTeX 데이터 저장이 완료되었습니다: {LATEX_DATA_FILE.name}")
        
        # 그래프 생성
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, far_rates * 100, label='FAR (False Acceptance Rate)', color='red')
        plt.plot(thresholds, frr_rates * 100, label='FRR (False Rejection Rate)', color='blue')
        
        plt.scatter(best_tau_p, best_eer * 100, color='green', zorder=5, s=100, label=f'EER Point ({best_eer * 100:.2f}%)')
        plt.axvline(x=best_tau_p, color='gray', linestyle='--', label=f'Optimal τ_p = {best_tau_p:.3f}')
        
        plt.title(f"Optimal FAR/FRR Curve ({final_result['mode']} Mode, $\\delta={best_weights[0]:.2f}, \\epsilon={best_weights[1]:.2f}$)")
        plt.xlabel('Threshold ($s_{p}$ score)'); plt.ylabel('Error Rate (%)'); plt.legend(); plt.grid(True, alpha=0.5); plt.ylim(bottom=0)
        plt.savefig(RESULT_PLOT_FILE)
        print(f"📈 최종 결과 그래프를 '{RESULT_PLOT_FILE.name}' 파일로 저장했습니다.")
    
    print("\n\n🔥 'NC\_SCALED'을 포함한 다섯 가지 공식에 대한 실험을 시작하세요. EER이 가장 낮은 공식을 채택하면 됩니다.")