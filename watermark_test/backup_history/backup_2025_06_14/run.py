import itertools, re, time, subprocess, sys, os
from pathlib import Path
from config import MODEL_DIR

ROOT_DIR   = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.py"
TRAIN_PY    = (ROOT_DIR / "model_experiment_final.py").resolve()
EVAL_PY     = (ROOT_DIR / "watermark_experiment_final2.py").resolve()
LOG_PATH    = ROOT_DIR / "results.txt"

GRID = {
    "LAM_Z"       : [0.05, 0.10],
    "LAM_J"       : [0.05, 0.08],
    "IMP_GAIN"    : [1.5, 2],
    "WM_STRENGTH" : [0.82, 0.84, 0.85],
    "BETA"        : [90],
    "WM_LEN"      : [256, 512, 1024],
    "WM_SEED"     : [42],
    "SCALE_LOGIT" : [45],
    "LAMBDA_CLEAN": [0.05, 0.08],
}

def patch_config(**pairs):
    txt = CONFIG_PATH.read_text(encoding="utf-8")
    for k, v in pairs.items():
        val = repr(v).replace("\\", "\\\\")      # ← 핵심 한 줄
        txt = re.sub(rf"^{k}\s*=.*$", f"{k} = {val}", txt, flags=re.M)
    CONFIG_PATH.write_text(txt, encoding="utf-8")

def run_one(combo: dict, run_idx: int):
    tag = "_".join(f"{k}{v}" for k, v in combo.items())
    print(f"\n▶▶ [{run_idx}] {tag}", flush=True)

    tag_folder = os.path.abspath(os.path.join(MODEL_DIR, tag))
    os.makedirs(tag_folder, exist_ok=True)

    patch_pairs = dict(combo)
    patch_pairs["MODEL_DIR"] = tag_folder        # ← 그냥 문자열!
    patch_config(**patch_pairs)

    t0 = time.time()
    subprocess.run([sys.executable, str(TRAIN_PY)], cwd=tag_folder, check=True)
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
            f.write(eval_out)
        else:
            f.write("[Evaluation Failed]\nSTDOUT:\n" + eval_out + "\nSTDERR:\n" + eval_err)
        f.write("\n")

    if success:
        print(f"✓ 저장 완료  (train {t_train/60:.1f} min / eval {t_eval:.1f} s)", flush=True)
    else:
        print("❌ 평가 실패:", tag)


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

