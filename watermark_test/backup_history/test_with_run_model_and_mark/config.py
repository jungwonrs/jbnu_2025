# ─── config.py ────────────────────────────────────────────────
import torch, os

# ▣ 파일/폴더 --------------------------------------------------
#for run2.py
#ROOT        = os.path.dirname(__file__)
ROOT        = os.path.join(os.path.dirname(__file__), "coco") 
TRAIN_JSON  = os.path.join(ROOT, r"annotations\instances_train2017.json")
TRAIN_DIR   = os.path.join(ROOT, "train2017")
MODEL_DIR = 'C:\\Users\\seo\\Desktop\\watermark_test\\LAM_Z0.5_LAM_J0.5_IMP_GAIN1_WM_STRENGTH0.05'

#for run2.py
#MODEL_DIR = r"C:\Users\seo\Desktop\watermark_test"
#init
#MODEL_DIR   = r"."

# ▣ 학습 -------------------------------------------------------
N_IMG       = 500  #10_000
BATCH       = 4
EPOCHS      = 50  #200
BLOCKS      = 12 #20
LR          = 1e-4
LAM_Z = 0.5
LAM_J = 0.5
IMP_TYPE      = 'grad'      # 'grad' | 'none'
IMP_SOBEL_K   = 3           # 3 or 5
IMP_REFRESH   = 10          # build_importance_map 호출 주기 (step)
IMP_GAIN = 1

# ▣ DWT / GPU --------------------------------------------------
WAVELET     = "haar"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ▣ 워터마크 ----------------------------------------------------
WM_STRING   = "Secret123"
WM_STRENGTH = 0.05
# ──────────────────────────────────────────────────────────────
