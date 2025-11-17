import torch
import os
import time
import random
import gc
from typing import Dict, Any
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights

from config import *
from record import add_result, save_to_excel
from attack_algo import sing_attack, hybrid_attack
from call_target_model import vml_model, classifier_model
from call_transfer_model import load_blip2_base, load_fuyu, load_llava13, load_clip_vit_l
from image_processor import (
    format_params, no_attack, jpeg_attack, blur_attack, noise_attack,
    calculate_metrics, load_coco_data, get_classifier_prediction,
    get_vlm_prediction, generate_attack_jobs
)

TRANSFER_LOADERS = {
    "BLIP-2": load_blip2_base,
    "Fuyu-8B": load_fuyu,
    "LLaVA-13B": load_llava13, 
    "CLIP-L-14": load_clip_vit_l,
}

IMAGENET_LABELS = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
def get_label_name(class_id):
    try: return f"{class_id} ({IMAGENET_LABELS[int(class_id)]})"
    except: return f"Class_{class_id}"

def tensor_to_pil(img_tensor):
    if img_tensor.dim() == 4: img_tensor = img_tensor.squeeze(0)
    img = (img_tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(img)

def is_attackable_classifier(source_name):
    return source_name in ["ResNet-50", "VGG16", "ViT-B/32"]

def preprocess_source_image(source_name, processor, img_pil):
    if source_name in ["ResNet-50", "VGG16"]:
        return processor(img_pil).unsqueeze(0).to(DEVICE)
    if source_name in ["ViT-B/32", "CLIP (ViT-B/32)"]:
        return processor(images=img_pil, return_tensors="pt")["pixel_values"].to(DEVICE)
    if source_name in ["LLaVA-v1.5-7B", "InstructBLIP"]:
        return T.ToTensor()(img_pil).unsqueeze(0).to(DEVICE)
    raise ValueError(f"Unknown: {source_name}")

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'): torch.cuda.ipc_collect()

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("./successful_attack_images", exist_ok=True)
    
    cleanup()
    coco_data, category_map = load_coco_data()
    if not coco_data: return

    ATTACK_CONFIG = [
    # ---- Single ----
    {"name": "FGSM", "func": sing_attack.fgsm_attack, "param_grid": FGSM_PARAM_GRID},
    {"name": "PGD", "func": sing_attack.pgd_attack, "param_grid": PGD_PARAM_GRID},
    {"name": "EOT_PGD_SINGLE", "func": sing_attack.eot_pgd_attack, "param_grid": EOT_PGD_PARAM_GRID},
    {"name": "BPDA_PGD_SINGLE", "func": sing_attack.bpda_pgd_attack, "param_grid": BPDA_PGD_PARAM_GRID},
    {"name": "CW", "func": sing_attack.cw_attack, "param_grid": CW_PARAM_GRID},
    {"name": "SPSA", "func": sing_attack.spsa_attack, "param_grid": SPSA_PARAM_GRID},

    # ---- Hybrid (10개 예시) ----
    {"name": "FGSM+PGD", "func": hybrid_attack.fgsm_then_pgd, "param_grid": FGSM_PGD_PARAM_GRID},
    {"name": "FGSM+SPSA", "func": hybrid_attack.fgsm_then_spsa, "param_grid": FGSM_SPSA_PARAM_GRID},
    {"name": "PGD+CW", "func": hybrid_attack.pgd_then_cw, "param_grid": PGD_CW_PARAM_GRID},
    {"name": "PGD+SPSA", "func": hybrid_attack.pgd_then_spsa, "param_grid": PGD_SPSA_PARAM_GRID},
    {"name": "FGSM+PGD+CW", "func": hybrid_attack.fgsm_pgd_cw, "param_grid": FGSM_PGD_CW_PARAM_GRID},
    {"name": "FGSM+PGD+SPSA", "func": hybrid_attack.fgsm_pgd_spsa, "param_grid": FGSM_PGD_SPSA_PARAM_GRID},
    {"name": "EOT+PGD", "func": hybrid_attack.eot_pgd, "param_grid": EOT_PGD_ONLY_PARAM_GRID},
    {"name": "BPDA+PGD", "func": hybrid_attack.bpda_pgd, "param_grid": BPDA_PGD_ONLY_PARAM_GRID},
    {"name": "EOT+PGD+CW", "func": hybrid_attack.eot_pgd_cw, "param_grid": EOT_PGD_CW_PARAM_GRID},
    {"name": "EOT+PGD+SPSA", "func": hybrid_attack.eot_pgd_spsa, "param_grid": EOT_PGD_SPSA_PARAM_GRID},
    ]

    ATTACK_JOBS = generate_attack_jobs(ATTACK_CONFIG)

    num_to_sample = min(NUM_IMAGES_TO_TEST, len(coco_data))
    coco_data_sample = random.sample(coco_data, num_to_sample)
    print(f"테스트 이미지 수: {len(coco_data_sample)}")

    source_models = {
        "LLaVA-v1.5-7B": vml_model.llava7,
        "InstructBLIP": vml_model.instructblip,
        "CLIP (ViT-B/32)": vml_model.clip_vit_b,
        "ViT-B/32": classifier_model.vit,
        "ResNet-50": classifier_model.resnet,
        "VGG16": classifier_model.vgg,
    }
    
    defense_dict = {"No attack": no_attack}
    if RUN_DEFENSE_COMPARISON:
        defense_dict.update({
            "JPEG 90%": lambda x: jpeg_attack(x, 90),
            "Blur 1.0": lambda x: blur_attack(x, 1.0),
            "Noise 0.01": lambda x: noise_attack(x, 0.01)
        })

    start_time = time.time()

    for source_name, source_loader in source_models.items():
        print(f"\n=== [Source] {source_name} ===")
        try:
            model_s, proc_s = source_loader()
        except Exception as e:
            print(f"Skip {source_name}: {e}"); continue

        is_classifier = is_attackable_classifier(source_name)
        is_vlm_source = source_name in ["LLaVA-v1.5-7B", "InstructBLIP"]

        for img_path, true_label_id, true_label_name in tqdm(coco_data_sample, desc=f"Processing {source_name}"):
            gt_label = true_label_name
            label_tensor = torch.tensor([true_label_id]).to(DEVICE)

            for defense_name, defense_func in defense_dict.items():
                cleanup()
                try:
                    img_pil = Image.open(img_path).convert("RGB")
                    img_tensor_orig = preprocess_source_image(source_name, proc_s, img_pil)
                    
                    if "JPEG" in defense_name: defended_img = defense_func(img_tensor_orig.cpu()).to(DEVICE)
                    else: defended_img = defense_func(img_tensor_orig).to(DEVICE)

                    initial_model_pred = "N/A"
                    safe_orig_id = -1
                    
                    if is_classifier:
                        safe_orig_id = get_classifier_prediction(model_s, defended_img)
                        initial_model_pred = get_label_name(safe_orig_id)
                    elif is_vlm_source:
                        pil_clean = tensor_to_pil(defended_img)
                        initial_model_pred = get_vlm_prediction(model_s, proc_s, pil_clean, gt_label, source_name)[:500]
                    elif "CLIP" in source_name:
                        pil_clean = tensor_to_pil(defended_img)
                        cands = list(category_map.values())
                        inputs = proc_s(text=[f"a photo of a {c}" for c in cands], images=pil_clean, return_tensors="pt", padding=True).to(DEVICE)
                        out = model_s(**inputs)
                        idx = out.logits_per_image.argmax().item()
                        initial_model_pred = f"Classified: {cands[idx]}"
                        
                except Exception as e:
                    print(f"Err Init: {e}"); continue

                adv_batch = []
                for job in ATTACK_JOBS:
                    if is_classifier:
                        try:
                            adv_img = job["func"](model_s, defended_img.clone(), label_tensor, **job["params"])
                            pred_adv_id = get_classifier_prediction(model_s, adv_img)
                            pred_adv_name = get_label_name(pred_adv_id)
                            success = (safe_orig_id != pred_adv_id)
                            psnr, ssim = calculate_metrics(defended_img, adv_img)

                            if success:
                                eps_str = f"{job['params']['epsilon']:.4f}"
                                safe_orig = str(gt_label).replace(" ", "_")
                                safe_pred = str(pred_adv_name).split("(")[0].strip()
                                fname = f"{source_name}_{job['name']}_eps{eps_str}_Orig({safe_orig})_Pred({safe_pred})_{os.path.basename(img_path)}"
                                save_path = os.path.join("./successful_attack_images", fname)
                                tensor_to_pil(adv_img).save(save_path)

                        except:
                            adv_img = defended_img.clone(); pred_adv_name="Err"; success=False; psnr=0; ssim=0
                    else:
                        adv_img = defended_img.clone()
                        if is_vlm_source:
                            pil_tmp = tensor_to_pil(adv_img)
                            vlm_out = get_vlm_prediction(model_s, proc_s, pil_tmp, gt_label, source_name)
                            pred_adv_name = vlm_out[:500]
                        elif "CLIP" in source_name:
                            pil_tmp = tensor_to_pil(adv_img)
                            cands = list(category_map.values())
                            inputs = proc_s(text=[f"a photo of a {c}" for c in cands], images=pil_tmp, return_tensors="pt", padding=True).to(DEVICE)
                            out = model_s(**inputs)
                            idx = out.logits_per_image.argmax().item()
                            pred_adv_name = f"Classified: {cands[idx]}"
                        else:
                            pred_adv_name = "Non-Generative"
                        
                        success=False; psnr=99.9; ssim=1.0
                    
                    adv_batch.append({
                        "img": adv_img.cpu(), "job": job, "p_name": pred_adv_name, 
                        "succ": success, "psnr": psnr, "ssim": ssim
                    })
                
                print(f"  [System] Clearing {source_name}...", end="\r")
                del model_s, proc_s
                cleanup()

                for trans_name, trans_loader in TRANSFER_LOADERS.items():
                    print(f"  -> [Transfer] {trans_name}...", end="\r")
                    model_t, proc_t = None, None
                    try:
                        model_t, proc_t = trans_loader()
                    except Exception as e:
                        print(f"\nSkip {trans_name}: {e}"); continue
                    
                    for item in adv_batch:
                        adv_gpu = item["img"].to(DEVICE)
                        pil_adv = tensor_to_pil(adv_gpu)
                        trans_res = "N/A"

                        try:
                            if "CLIP" in trans_name:
                                cands = list(category_map.values())
                                inputs = proc_t(text=[f"a photo of a {c}" for c in cands], images=pil_adv, return_tensors="pt", padding=True).to(DEVICE)
                                out = model_t(**inputs)
                                idx = out.logits_per_image.argmax().item()
                                trans_res = f"Classified: {cands[idx]}"
                            elif hasattr(model_t, "generate"):
                                trans_res = get_vlm_prediction(model_t, proc_t, pil_adv, gt_label, trans_name)
                        except Exception as e:
                            trans_res = f"Err: {e}"
                        
                        add_result({
                            "file_name": os.path.basename(img_path),
                            "image_attack": defense_name,
                            "target_model": source_name,
                            "image_label_coco": gt_label,
                            "image_label_model": initial_model_pred,
                            "after_evasion_label": item["p_name"],
                            "evasion_attack_algo": item["job"]["name"],
                            **format_params(item["job"]["params"]),
                            "target_model_attack_results": item["succ"],
                            "PSNR": item["psnr"], "SSIM": item["ssim"],
                            "transfer_model": trans_name,
                            "transfer_model_label": trans_res,
                        })
                        del adv_gpu

                    del model_t, proc_t
                    cleanup()
                
                try:
                    model_s, proc_s = source_loader()
                except:
                    pass 

        if 'model_s' in locals(): del model_s
        if 'proc_s' in locals(): del proc_s
        cleanup()

    print(f"\nDone. Time: {(time.time()-start_time)/60:.1f}m")
    save_to_excel()

if __name__ == "__main__":
    main()