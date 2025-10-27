import itertools, re, time, subprocess, sys, os
from pathlib import Path
from config import MODEL_DIR

ROOT_DIR   = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.py"
TRAIN_PY    = (ROOT_DIR / "model.py").resolve()
EVAL_PY     = (ROOT_DIR / "watermark_experiment.py").resolve()
LOG_PATH    = ROOT_DIR / "results.txt"


'''done-f_test_1
#-----------best(3개짜리)--------
GRID = {
    "LAM_Z":        [0.07],        
    "LAM_J":        [0.07],             
    "IMP_GAIN":     [1.35],        
    "LOGIT_COEFF":  [110],            
    "WM_LEN":       [256], 
    "WM_SEED":      [42],
    "BLOCKS":       [10],
    "EPOCHS":       [200],                
    "BATCH":        [8], 
    "LAMBDA_DISTORTION": [250],  # 높을수록 PSNR 향상
    "BETA_MAX":          [150],  # 높을수록 강인성 향상
    "WM_STRENGTH":       [10.0],       # 높을수록 강인성 향상
    "MAG_WEIGHT": [0.2],              
}
'''
'''
#-----------Performance(3개짜리)-----f_test_2.2---
GRID = {
    "LAM_Z":        [0.01],        
    "LAM_J":        [0.01],             
    "IMP_GAIN":     [3.0],        
    "LOGIT_COEFF":  [250],            
    "WM_LEN":       [256], 
    "WM_SEED":      [42],
    "BLOCKS":       [10],
    "EPOCHS":       [200],                
    "BATCH":        [8], 
    "LAMBDA_DISTORTION": [250],  # 높을수록 PSNR 향상
    "BETA_MAX":          [150],  # 높을수록 강인성 향상
    "WM_STRENGTH":       [10.0],       # 높을수록 강인성 향상
    "MAG_WEIGHT": [0.2],              
}
'''
'''
#-----------Performance(3개짜리)-----f_test_2.3---
GRID = {
    "LAM_Z":        [0.01],        
    "LAM_J":        [0.5],             
    "IMP_GAIN":     [0.5, 3.0],        
    "LOGIT_COEFF":  [50, 250],            
    "WM_LEN":       [256], 
    "WM_SEED":      [42],
    "BLOCKS":       [10],
    "EPOCHS":       [200],                
    "BATCH":        [8], 
    "LAMBDA_DISTORTION": [250],  # 높을수록 PSNR 향상
    "BETA_MAX":          [150],  # 높을수록 강인성 향상
    "WM_STRENGTH":       [10.0],       # 높을수록 강인성 향상
    "MAG_WEIGHT": [0.2],              
}
'''
'''
#-----------Performance(3개짜리)-----f_test_2.4---
GRID = {
    "LAM_Z":        [0.5],        
    "LAM_J":        [0.01, 0.5],             
    "IMP_GAIN":     [0.5, 3.0],        
    "LOGIT_COEFF":  [50, 250],            
    "WM_LEN":       [256], 
    "WM_SEED":      [42],
    "BLOCKS":       [10],
    "EPOCHS":       [200],                
    "BATCH":        [8], 
    "LAMBDA_DISTORTION": [250],  # 높을수록 PSNR 향상
    "BETA_MAX":          [150],  # 높을수록 강인성 향상
    "WM_STRENGTH":       [10.0],       # 높을수록 강인성 향상
    "MAG_WEIGHT": [0.2],              
}
'''


'''
#-----------Peformance: Robustness (3개짜리)-----f_test_3---
GRID = {
    "LAM_Z":        [0.07],        
    "LAM_J":        [0.07],             
    "IMP_GAIN":     [1.35],        
    "LOGIT_COEFF":  [110],            
    "WM_LEN":       [256], 
    "WM_SEED":      [42],
    "BLOCKS":       [10],
    "EPOCHS":       [200],                
    "BATCH":        [8], 
    "LAMBDA_DISTORTION": [150], 
    "BETA_MAX":          [300],
    "WM_STRENGTH":       [15.0],       
    "MAG_WEIGHT": [0.2],              
}
'''

'''
#-----------Peformance: PSNR (3개짜리)------f_test_4--
GRID = {
    "LAM_Z":        [0.07],        
    "LAM_J":        [0.07],             
    "IMP_GAIN":     [1.35],        
    "LOGIT_COEFF":  [110],            
    "WM_LEN":       [256], 
    "WM_SEED":      [42],
    "BLOCKS":       [10],
    "EPOCHS":       [200],                
    "BATCH":        [8], 
    "LAMBDA_DISTORTION": [500], 
    "BETA_MAX":          [100],
    "WM_STRENGTH":       [7.0],       
    "MAG_WEIGHT": [0.2],              
}
'''


'''
#-----------Ablation Remove LAMBDA_DISTORTION (3개짜리 하고 1개 건지기)-----f_test_5---
GRID = {
    "LAM_Z":        [0.07],        
    "LAM_J":        [0.07],             
    "IMP_GAIN":     [1.35],        
    "LOGIT_COEFF":  [110],            
    "WM_LEN":       [256], 
    "WM_SEED":      [42],
    "BLOCKS":       [10],
    "EPOCHS":       [200],                
    "BATCH":        [8], 
    "LAMBDA_DISTORTION": [0], 
    "BETA_MAX":          [150],
    "WM_STRENGTH":       [10.0],       
    "MAG_WEIGHT": [0.2],              
}
'''


'''
#-----------Ablation Remove MAG_WEIGHT  (3개짜리 하고 1개 건지기)----f_test_6----
GRID = {
    "LAM_Z":        [0.07],        
    "LAM_J":        [0.07],             
    "IMP_GAIN":     [1.35],        
    "LOGIT_COEFF":  [110],            
    "WM_LEN":       [256], 
    "WM_SEED":      [42],
    "BLOCKS":       [10],
    "EPOCHS":       [200],                
    "BATCH":        [8], 
    "LAMBDA_DISTORTION": [0, 250], 
    "BETA_MAX":          [150],
    "WM_STRENGTH":       [10.0],       
    "MAG_WEIGHT": [0.0],              
}

'''


#-----------best(3개짜리) with block, epoch, batch..----f_test_7----
GRID = {
    "LAM_Z":        [0.07],        
    "LAM_J":        [0.07],             
    "IMP_GAIN":     [1.35],        
    "LOGIT_COEFF":  [110],            
    "WM_LEN":       [256], 
    "WM_SEED":      [42],
    "BLOCKS":       [10, 20, 25],
    "EPOCHS":       [200, 300],                
    "BATCH":        [8, 16], 
    "LAMBDA_DISTORTION": [250], 
    "BETA_MAX":          [150],
    "WM_STRENGTH":       [10.0],       
    "MAG_WEIGHT": [0.2],              
}



def patch_config(**pairs):
    txt = CONFIG_PATH.read_text(encoding="utf-8")
    for k, v in pairs.items():
        val = repr(v).replace("\\", "\\\\")     
        txt = re.sub(rf"^{k}\s*=.*$", f"{k} = {val}", txt, flags=re.M)
    CONFIG_PATH.write_text(txt, encoding="utf-8")

def run_one(combo: dict, run_idx: int):
    epochs = combo["EPOCHS"]
    combo["EPOCHS_A"] = max(1, epochs // 4)      
    combo["EPOCHS_B"] = epochs - combo["EPOCHS_A"]

    # 로그에 기록될 긴 태그는 그대로 유지합니다.
    tag = "_".join(f"{k}{v}" for k, v in combo.items())
    print(f"\n▶▶ [{run_idx}] {tag}", flush=True)

    # [수정] 폴더 이름은 run_1, run_2 처럼 매우 짧게 만듭니다.
    run_folder_name = f"run_{run_idx}"
    # [수정] MODEL_DIR의 기본 경로를 현재 스크립트 위치 아래 'models' 폴더로 변경합니다.
    base_model_dir = os.path.join(ROOT_DIR, 'models')
    tag_folder = os.path.abspath(os.path.join(base_model_dir, run_folder_name))
    
    os.makedirs(tag_folder, exist_ok=True)

    patch_pairs = dict(combo)
    # config.py에 기록될 MODEL_DIR 경로는 방금 만든 짧은 폴더 경로를 사용합니다.
    patch_pairs["MODEL_DIR"] = tag_folder      
    patch_config(**patch_pairs)

    t0 = time.time()
    # [수정] capture_output=True를 삭제하여 로그가 실시간으로 터미널에 출력되도록 합니다.
    try:
        subprocess.run([sys.executable, str(TRAIN_PY)], cwd=tag_folder, check=True)
    except subprocess.CalledProcessError as e:
        # 훈련 중 에러가 발생하면 여기서 감지됩니다.
        print(f"❌ 훈련 실패: run_{run_idx}")
        with LOG_PATH.open("a", encoding="utf8") as f:
            f.write(f"\n========== Run {run_idx}: {tag} ==========\n")
            f.write(f"[Training Failed with exit code {e.returncode}]\n")
        return # 다음 실험으로 넘어갑니다.

    t_train = time.time() - t0

    # -------- 평가 --------
    t0 = time.time()
    try:
        eval_proc = subprocess.run(
            [sys.executable, str(EVAL_PY), tag],
            cwd=tag_folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=False, check=True
        )
        eval_out = eval_proc.stdout.decode("utf-8", errors="replace")
        eval_err = eval_proc.stderr.decode("utf-8", errors="replace")
        success  = True
    except subprocess.CalledProcessError as e:
        eval_out = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
        eval_err = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        success  = False
    t_eval = time.time() - t0

    # -------- 로그 기록 --------
    with LOG_PATH.open("a", encoding="utf8") as f:
        f.write(f"\n========== Run {run_idx}: {tag} ==========\n")
        f.write(f"Train time: {t_train/60:.1f} min | Eval time: {t_eval:.1f} s\n")
        if success:
            marker = "\n===== AVERAGE OVER ALL IMAGES ====="
            f.write(eval_out[eval_out.find(marker):] if marker in eval_out else eval_out)
        else:
            f.write("[Evaluation Failed]\nSTDOUT:\n" + eval_out + "\nSTDERR:\n" + eval_err)
        f.write("\n")

    if success:
        print(f"✓ 저장 완료  (train {t_train/60:.1f} min / eval {t_eval:.1f} s)", flush=True)
    else:
        print(f"❌ 평가 실패: run_{run_idx}")

# ── main ───────────────────────────────────────────────────
if __name__ == "__main__":
    LOG_PATH.write_text("### Automated INN Watermark experiment log ###\n")
    combos = list(itertools.product(*GRID.values()))
    keys   = list(GRID.keys())

    t_all = time.time()
    for i, vals in enumerate(combos, 1):
        run_one(dict(zip(keys, vals)), i)

    print(f"\n◎ 모든 실험 종료 - 총 소요 {(time.time() - t_all) / 60:.1f}분", flush=True)
# ───────────────────────────────────────────────────────────
