import itertools, re, time, subprocess, fileinput, sys, json, os
from pathlib import Path
from config import MODEL_DIR

CONFIG_PATH = Path("config.py")
TRAIN_PY = "model_experiment_final.py"
EVAL_PY = "watermark_experiment_final2.py"
LOG_PATH = Path("results.txt")


GRID = {
    "LAM_Z" : [0.5, 0.1],
    "LAM_J" : [0.5, 0.1],
    "IMP_GAIN" : [1, 3, 5],
    "WM_STRENGTH" : [0.05, 0.1, 0.2],
}


def patch_config(**pairs):
    txt = CONFIG_PATH.read_text(encoding='utf-8')
    for k, v in pairs.items():
        txt = re.sub(rf"^{k}\s*=.*$", f"{k} = {v}", txt, flags=re.M)
    CONFIG_PATH.write_text(txt, encoding='utf-8')


def run_one(combo: dict, run_idx: int):
    tag = "_".join(f"{k}{v}" for k, v in combo.items())
    print(f"\n▶▶ [{run_idx}] {tag}")

    # 1) config.py 패치
    patch_config(**combo)

    tag_folder = os.path.join(MODEL_DIR, tag)
    os.makedirs(tag_folder, exist_ok=True)

    # 2) 학습
    t0 = time.time()
    script_train = os.path.join(os.getcwd(), TRAIN_PY)
    subprocess.run([sys.executable, script_train], cwd=tag_folder, check=True)
    t_train = time.time() - t0

    # 3) 평가
    t0 = time.time()
    script_eval = os.path.join(os.getcwd(), EVAL_PY)

    try:
        # 바이너리로 받아 수동 디코딩 (깨짐 방지)
        eval_proc = subprocess.run(
            [sys.executable, script_eval, tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            check=True
        )
        # UTF-8로 디코딩 시도, 실패 시 대체문자 사용
        eval_out = eval_proc.stdout.decode("utf-8", errors="replace")
        eval_err = eval_proc.stderr.decode("utf-8", errors="replace")
        success = True

    except subprocess.CalledProcessError as e:
        eval_out = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
        eval_err = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        success = False

    t_eval = time.time() - t0

    # 4) 결과 로그
    with LOG_PATH.open("a", encoding="utf8") as f:
        f.write(f"\n========== Run {run_idx}: {tag} ==========\n")
        f.write(f"Train time: {t_train/60:.1f} min | Eval time: {t_eval:.1f} s\n")
        if success:
            f.write(eval_out)
        else:
            f.write("[Evaluation Failed]\n")
            f.write("STDOUT:\n" + eval_out)
            f.write("\nSTDERR:\n" + eval_err)
        f.write("\n")

    if success:
        print(f"✓ 저장 완료  (train {t_train/60:.1f} min  / eval {t_eval:.1f} s)")
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

    print(f"\n◎ 모든 실험 종료 - 총 소요 {(time.time() - t_all) / 60:.1f}분")
# ───────────────────────────────────────────────────────────

