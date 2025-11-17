import os
import torch

HF_TOKEN = "-" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
print(f"Using DType: {DTYPE}")

RUN_DEFENSE_COMPARISON = False

# image set
NUM_IMAGES_TO_TEST = 1
#NUM_IMAGES_TO_TEST = 3000

# image dir
COCO_ROOT = "coco"
COCO_VAL_IMAGES = os.path.join(COCO_ROOT, "val2017")
COCO_ANNOTATIONS = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")

# result dir
RESULTS_DIR = "results"
ATTACK_IMAGE_DIR = os.path.join(RESULTS_DIR, "successful_attack_images")
FINAL_EXCEL_PATH = os.path.join(RESULTS_DIR, "final_experiment_results.xlsx")

# excel setting
EXCEL_COLUMNS = [
    "file_name", "image_attack", "target_model", "image_label_coco", "image_label_model", "evasion_attack_algo", 
    "algo_para_1", "algo_para_2", "algo_para_3", "algo_para_4", 
    "algo_para_5", "algo_para_6", "algo_para_7", "algo_para_8",
    "algo_para_9", "algo_para_10",
    "after_evasion_label", "target_model_attack_results", "PSNR", "SSIM", 
    "transfer_model", "transfer_model_label", "transfer_model_attack_results"
]


# --- Single attacks ---
FGSM_PARAM_GRID = {
    "epsilon": [1/255, 2/255, 4/255, 8/255, 16/255],
}

PGD_PARAM_GRID = {
    "epsilon": [4/255, 8/255],
    "alpha": [1/255, 2/255],
    "num_steps": [10],
    "random_start": [True],
}

EOT_PGD_PARAM_GRID = {
    "epsilon": [4/255],
    "alpha": [1/255],
    "num_steps": [10],
    "eot_samples": [5],
    "eot_sigma": [0.01],
}

BPDA_PGD_PARAM_GRID = {
    "epsilon": [4/255],
    "alpha": [1/255],
    "num_steps": [10],
    # defense_fn 은 코드에서 직접 넣어야 해서 여기서는 생략 (or None)
}

CW_PARAM_GRID = {
    "c": [1.0],
    "kappa": [0.0],
    "num_steps": [50],
    "lr": [0.01],
}

SPSA_PARAM_GRID = {
    "epsilon": [4/255],
    "num_steps": [20],
    "spsa_samples": [64],
    "spsa_delta": [0.01],
    "step_size": [1/255],
}

# --- Hybrid attacks (예시 10개) ---
FGSM_PGD_PARAM_GRID = {
    "epsilon": [4/255],
    "pgd_alpha": [1/255],
    "pgd_steps": [10],
    "pgd_random_start": [True],
}

FGSM_SPSA_PARAM_GRID = {
    "epsilon": [4/255],
    "spsa_steps": [10],
    "spsa_samples": [64],
    "spsa_delta": [0.01],
    "spsa_step_size": [1/255],
}

PGD_CW_PARAM_GRID = {
    "epsilon": [4/255],
    "pgd_alpha": [1/255],
    "pgd_steps": [10],
    "cw_c": [1.0],
    "cw_kappa": [0.0],
    "cw_steps": [50],
    "cw_lr": [0.01],
}

PGD_SPSA_PARAM_GRID = {
    "epsilon": [4/255],
    "pgd_alpha": [1/255],
    "pgd_steps": [10],
    "spsa_steps": [10],
    "spsa_samples": [64],
    "spsa_delta": [0.01],
    "spsa_step_size": [1/255],
}

FGSM_PGD_CW_PARAM_GRID = {
    "epsilon": [4/255],
    "pgd_alpha": [1/255],
    "pgd_steps": [10],
    "cw_c": [1.0],
    "cw_kappa": [0.0],
    "cw_steps": [50],
    "cw_lr": [0.01],
}

FGSM_PGD_SPSA_PARAM_GRID = {
    "epsilon": [4/255],
    "pgd_alpha": [1/255],
    "pgd_steps": [10],
    "spsa_steps": [10],
    "spsa_samples": [64],
    "spsa_delta": [0.01],
    "spsa_step_size": [1/255],
}

EOT_PGD_ONLY_PARAM_GRID = {
    "epsilon": [4/255],
    "alpha": [1/255],
    "num_steps": [10],
    "eot_samples": [5],
    "eot_sigma": [0.01],
}

BPDA_PGD_ONLY_PARAM_GRID = {
    "epsilon": [4/255],
    "alpha": [1/255],
    "num_steps": [10],
    # defense_fn 은 코드에서 직접 넣기
}

EOT_PGD_CW_PARAM_GRID = {
    "epsilon": [4/255],
    "alpha": [1/255],
    "num_steps": [10],
    "eot_samples": [5],
    "eot_sigma": [0.01],
    "cw_c": [1.0],
    "cw_kappa": [0.0],
    "cw_steps": [50],
    "cw_lr": [0.01],
}

EOT_PGD_SPSA_PARAM_GRID = {
    "epsilon": [4/255],
    "alpha": [1/255],
    "num_steps": [10],
    "eot_samples": [5],
    "eot_sigma": [0.01],
    "spsa_steps": [10],
    "spsa_samples": [64],
    "spsa_delta": [0.01],
    "spsa_step_size": [1/255],
}