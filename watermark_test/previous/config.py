# ─── config.py ────────────────────────────────────────────────
import torch, os

# ▣ 파일/폴더 --------------------------------------------------
ROOT        = r"C:\Users\seo\Desktop\watermark_test\coco"
TRAIN_JSON  = os.path.join(ROOT, r"annotations\instances_train2017.json")
TRAIN_DIR   = os.path.join(ROOT, "train2017")
MODEL_DIR   = r"."

# ▣ 학습 -------------------------------------------------------
N_IMG       = 2_000
BATCH       = 4
EPOCHS      = 10
BLOCKS      = 20
LR          = 1e-4
LAM_Z       = 0.5          # loss 가중치
LAM_J       = 1.0
IMP_TYPE      = 'grad'      # 'grad' | 'none'
IMP_SOBEL_K   = 3           # 3 or 5
IMP_REFRESH   = 10          # build_importance_map 호출 주기 (step)
IMP_GAIN      = 5.0         # loss =  LAM_Z * IMP_GAIN * loss_img

# ▣ DWT / GPU --------------------------------------------------
WAVELET     = "haar"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ▣ 워터마크 ----------------------------------------------------
WM_STRING   = "Secret123"
WM_STRENGTH = 0.05         # ± 값
# ──────────────────────────────────────────────────────────────
