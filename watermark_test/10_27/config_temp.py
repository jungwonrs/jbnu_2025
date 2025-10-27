import torch, os
ROOT        = os.path.join(os.path.dirname(__file__), "coco") 
TRAIN_JSON  = os.path.join(ROOT, r"annotations\instances_train2017.json")
TRAIN_DIR   = os.path.join(ROOT, "train2017")
TEST_DIR    = os.path.join(os.path.dirname(__file__), "images")
MODEL_DIR = 'C:\\Users\\seo\\Desktop\\watermark_experiment\\models\\run_2'
MODEL_DIR2 = 'C:\\Users\\seo\\Desktop\\watermark_experiment\\models2\\run_1'

#MODEL_DIR = r'.'


# model settings
N_IMG = 10000
#N_IMG = 1000
BLOCKS = 10
BATCH = 8
EPOCHS = 200
EPOCHS_A = 50
EPOCHS_B = 150
LR          = 1e-5
LAM_Z = 0.07
LAM_J = 0.07
BETA = 100
IMP_GAIN = 1.35
LAMBDA_CLEAN = 0.08
BETA_MAX = 150
BETA_FLOOR = 20
LOGIT_COEFF = 110
LAMBDA_DISTORTION = 0
MAG_WEIGHT = 0.0

# watermark settings 
WM_STRENGTH = 10.0
WM_LEN = 256
WM_SEED = 42

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
GN_SIGMA = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
