import torch, os
ROOT        = os.path.join(os.path.dirname(__file__), "coco") 
TRAIN_JSON  = os.path.join(ROOT, r"annotations\instances_train2017.json")
TRAIN_DIR   = os.path.join(ROOT, "train2017")
TEST_DIR    = os.path.join(os.path.dirname(__file__), "images")
MODEL_DIR = 'C:\\Users\\seo\\Desktop\\watermark_experiment\\LAM_Z0.07_LAM_J0.04_IMP_GAIN1.46_WM_STRENGTH0.93_BETA90_WM_LEN256_WM_SEED42_SCALE_LOGIT43_LAMBDA_CLEAN0.08_BLOCKS6_EPOCHS30_EPOCHS_A10_EPOCHS_B20'

#MODEL_DIR = r'.'


# model settings
N_IMG = 10000
BLOCKS = 6
BATCH = 4
EPOCHS = 30
EPOCHS_A = 10
EPOCHS_B = 20
LR          = 1e-4
LAM_Z = 0.07
LAM_J = 0.04
BETA = 90
IMP_GAIN = 1.46
LAMBDA_CLEAN = 0.08

# watermark settings 
WM_STRENGTH = 0.93
WM_LEN = 256
WM_SEED = 42
SCALE_LOGIT = 43

# util settings
WAVELET     = "haar"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# watermark embedding & extracting
TEST_N_IMG = 1000

# attack config
'''
JPEG_Q = [70]
GB_SIG = [2.0]
RS_SCALES = [0.7]
CR_PCTS = [0.8]
GN_SIGMA = [0.03]
'''

JPEG_Q = [50, 60, 70, 80, 90]
GB_SIG = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
RS_SCALES = [0.5, 0.6, 0.7, 0.8, 0.9]
CR_PCTS = [0.6, 0.7, 0.8, 0.9]
GN_SIGMA = [0.01, 0.02, 0.03, 0.05, 0.06, 0.07]
