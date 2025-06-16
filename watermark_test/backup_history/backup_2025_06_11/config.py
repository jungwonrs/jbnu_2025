# ─── config.py ────────────────────────────────────────────────
import torch, os

# ▣ 파일/폴더 --------------------------------------------------
#for run2.py
#ROOT        = os.path.dirname(__file__)
ROOT        = os.path.join(os.path.dirname(__file__), "coco") 
TRAIN_JSON  = os.path.join(ROOT, r"annotations\instances_train2017.json")
TRAIN_DIR   = os.path.join(ROOT, "train2017")
MODEL_DIR = 'C:\\Users\\seo\\Desktop\\watermark_test\\LAM_Z0.2_LAM_J0.05_IMP_GAIN2_WM_STRENGTH0.75_BETA60_WM_LEN256_WM_SEED42_SCALE_LOGIT60'

#for run2.py
#MODEL_DIR = r"C:\Users\seo\Desktop\watermark_test"
#init
#MODEL_DIR   = r"."

# ▣ 학습 -------------------------------------------------------
N_IMG       = 2000  #10_000
BATCH       = 4
EPOCHS      = 30  #200
BLOCKS      = 8 #20
LR          = 2e-4
LAM_Z = 0.2
LAM_J = 0.05
IMP_TYPE      = 'grad'      # 'grad' | 'none'
IMP_SOBEL_K   = 3           # 3 or 5
IMP_REFRESH   = 10          # build_importance_map 호출 주기 (step)
IMP_GAIN = 2
EPOCHS_A = 15 #이미지 품질 위주 PRE-TRAIN 단계계
EPOCHS_B = EPOCHS -EPOCHS_A # 워터마크 복원 단계
BETA = 60

# ▣ DWT / GPU --------------------------------------------------
WAVELET     = "haar"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ▣ 워터마크 ----------------------------------------------------
WM_STRING   = "Secret123"
WM_STRENGTH = 0.75
WM_LEN = 256
WM_SEED = 42

SCALE_LOGIT = 60
# ──────────────────────────────────────────────────────────────
