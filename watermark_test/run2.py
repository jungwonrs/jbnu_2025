# ───────── run2.py ─────────
"""
· MODEL_DIR 하위의 모든 실험 폴더
    ├─ both/inn_both.pth
    ├─ lh/inn_lh.pth
    ├─ hl/inn_hl.pth
    └─ full/inn_full.pth
  을 가진 경우에만 평가 스크립트(watermark_experiment_final2.py) 실행
· 각 실험 STDOUT/STDERR를 한 파일(all_results.txt)에 계속 append
"""

import os, re, sys, time, subprocess
from pathlib import Path

CONFIG_PATH   = Path("config.py")
orig_cfg_text = CONFIG_PATH.read_text(encoding="utf-8")
COMMON_LOG    = Path("all_results.txt")

def patch_model_dir(new_dir: str):
    escaped = new_dir.replace("\\","\\\\")
    new_line = f'MODEL_DIR = r"{escaped}"'
    cfg = re.sub(r"^MODEL_DIR\s*=.*$", new_line, orig_cfg_text, flags=re.M)
    CONFIG_PATH.write_text(cfg, encoding="utf-8")

def run_one(exp_dir: Path):
    tag = exp_dir.name
    print(f"\n▶▶ Evaluating '{tag}'")
    patch_model_dir(str(exp_dir.resolve()))
    t0 = time.time()
    proc = subprocess.run([sys.executable,"watermark_experiment_final2.py"],
                          capture_output=True, text=True)
    dt = time.time()-t0
    with COMMON_LOG.open("a",encoding="utf-8") as fp:
        fp.write(f"\n========== {tag} ==========\n")
        fp.write(f"Start  : {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(t0))}\n")
        fp.write(f"Elapsed: {dt:.1f} s\n\n")
        fp.write(proc.stdout+("\n"+proc.stderr if proc.stderr else "")+"\n")
    print(f"  ✓ done ({dt:.1f}s) → logged")

if __name__=="__main__":
    from config import MODEL_DIR
    root = Path(MODEL_DIR)
    if not root.is_dir():
        print(f"MODEL_DIR '{root}' is invalid"); sys.exit(1)

    COMMON_LOG.write_text(
        "### INN Watermark Attack Evaluation Log ###\n"
        f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")

    for exp in sorted(root.iterdir()):
        if not exp.is_dir(): continue
        if all((exp/p).exists() for p in [
                Path("both/inn_both.pth"),
                Path("lh/inn_lh.pth"),
                Path("hl/inn_hl.pth"),
                Path("full/inn_full.pth")]):
            run_one(exp)
        else:
            print(f"[-] skip '{exp.name}' (missing .pth files)")

    CONFIG_PATH.write_text(orig_cfg_text, encoding="utf-8")
    print("\n◎ All evaluations done — config.py restored.")
