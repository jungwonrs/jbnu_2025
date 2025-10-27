# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import pywt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm
import random
import sys

# FrEIA 프레임워크 임포트 (설치 필요: pip install FrEIA)
try:
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm
except ImportError:
    print("❌ ERROR: 'FrEIA' 라이브러리를 찾을 수 없습니다.")
    print("라이브러리를 설치해주세요: pip install FrEIA")
    sys.exit(1)


# ==================================================================================== #
# ✨ 1부: 독립 실행을 위한 유틸리티 및 상수 ✨ #
# ==================================================================================== #

# -------------------------- #
#  실험 설정값 (Constants)
# -------------------------- #
BLOCKS = 10
WM_STRENGTH = 10.0
WM_LEN = 256
WM_SEED = 42
WAVELET = "haar"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

JPEG_Q = [90, 80, 70, 60, 50] # 90 -> 50 순서
GB_SIG = [0.5]
GN_SIGMA = [0.01, 0.02, 0.03] 

# -------------------------- #
#  모델 아키텍처 (Model Architecture)
# -------------------------- #
def subnet(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Conv2d(c_in, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, c_out, 3, padding=1)
    )

def build_inn(C, H, W, blocks):
    nodes = [Ff.InputNode(C, H, W, name="in")]
    for k in range(blocks):
        nodes.append(Ff.Node(nodes[-1], Fm.AllInOneBlock, {"subnet_constructor": subnet}, name=f"inn_{k}"))
    nodes.append(Ff.OutputNode(nodes[-1], name="out"))
    return Ff.ReversibleGraphNet(nodes, verbose=False)

def load_net(pth):
    print(f"모델 파일 로드 시도: {pth}")
    if not os.path.exists(pth):
        print(f"❌ ERROR: 모델 파일을 찾을 수 없습니다! 경로를 확인하세요.")
        print(f"시도한 경로: {pth}")
        sys.exit(1)
    
    try:
        ck = torch.load(pth, map_location=DEVICE)
        C, H, W = ck["in_shape"]
        net = build_inn(C, H, W, ck["num_blocks"]).to(DEVICE)
        net.load_state_dict(ck["state_dict"])
        net.eval()
        print("✅ 모델 로드 성공.")
        return net
    except Exception as e:
        print(f"❌ ERROR: 모델 로드 중 오류 발생: {e}")
        sys.exit(1)

# ------------------------------- #
#  워터마크 생성/삽입 (watermark_experiment.py 기반)
# ------------------------------- #
def tile_np(arr):
    flat = np.repeat(arr, (128*128 + len(arr)-1)//len(arr))[:128*128]
    return flat.reshape(128, 128).astype(np.float32)

def make_watermark():
    rng = np.random.RandomState(WM_SEED)
    bits = rng.randint(0, 2, WM_LEN, dtype=np.uint8)
    mid = WM_LEN // 2
    bitsA, bitsB = bits[:mid], bits[mid:]
    mapA, mapB = tile_np(bitsA), tile_np(bitsB)
    return mapA, mapB

@torch.no_grad()
def add_wm_split(z, chA, chB, mapA, mapB):
    wm_tA = torch.tensor((mapA*2-1) * WM_STRENGTH, device=DEVICE)  
    wm_tB = torch.tensor((mapB*2-1) * WM_STRENGTH, device=DEVICE)
    z2 = z.clone()
    z2[:, chA] = z2[:, chA] + wm_tA
    z2[:, chB] = z2[:, chB] + wm_tB
    return z2

# ------------------------------- #
#  이미지 처리 함수 (Image Processing)
# ------------------------------- #
def coeff2img(coeff, gray):
    if isinstance(coeff, tuple): coeff = coeff[0]
    
    if coeff.shape[1] == 2:       # 2채널 (LH, HL)
        LL_ = pywt.dwt2(gray/255., WAVELET)[0] 
        LH_,HL_ = coeff[0,0].cpu().numpy(), coeff[0,1].cpu().numpy()
        HH_ = np.zeros_like(LH_) 
    elif coeff.shape[1] == 4:     # 4채널 (LL, LH, HL, HH)
        LL_,LH_,HL_,HH_ = [c.cpu().numpy() for c in coeff[0]]
    else:
        raise ValueError(f"예상치 못한 계수 채널 수: {coeff.shape[1]}")
    
    return pywt.idwt2((LL_,(LH_,HL_,HH_)), WAVELET)

# -------------------------- #
#  공격 함수 (Attack Functions)
# -------------------------- #
def jpeg(u8, q):
    enc = cv2.imencode('.jpg', u8, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

def gblur(f, s):
    k = int(s * 4 + 1) | 1
    return cv2.GaussianBlur(f, (k, k), sigmaX=s, borderType=cv2.BORDER_REPLICATE)

def add_noise(u8, sigma):
    f = u8.astype(np.float32) / 255.
    noisy = np.clip(f + np.random.randn(*f.shape) * sigma, 0, 1)
    return (noisy * 255).round().astype(np.uint8)


# ==================================================================================== #
# ✨ 2부: 메인 시각화 로직 ✨ #
# ==================================================================================== #

MODEL_FILE_PATH = Path("C:\\Users\\seo\\Desktop\\watermark_experiment\\visual_folder\\inn_both.pth")
TEST_DIR = Path(__file__).resolve().parent / "images"
NUM_IMAGES_TO_SHOW = 2
DIFF_MAGNIFICATION = 3 # 시각적 강조 배율
OUTPUT_FILENAME = "stego_paper_layout_FINAL_LABELS.png" # 출력 파일명

# -------------------------- #
#  핵심 기능 함수
# -------------------------- #
def load_sample_images(image_dir, num_images):
    if not image_dir.exists():
        print(f"❌ ERROR: 테스트 이미지 폴더를 찾을 수 없습니다: '{image_dir}'")
        sys.exit(1)
    all_images = list(image_dir.glob("*.jpg"))
    if len(all_images) == 0:
        print(f"❌ ERROR: '{image_dir}' 폴더에서 .jpg 이미지를 찾을 수 없습니다.")
        sys.exit(1)
    if len(all_images) < num_images:
        print(f"⚠️  WARNING: 이미지가 {num_images}장보다 적습니다. ({len(all_images)}장만 사용)")
        num_images = len(all_images)

    selected_paths = random.sample(all_images, num_images)
    results = []
    print(f"🖼️  {num_images}개의 샘플 이미지를 불러옵니다 (흑백으로 변환)...")
    for image_path in selected_paths:
        img_color = cv2.imread(str(image_path))
        img_resized = cv2.resize(img_color, (256, 256))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        LL, (LH, HL, HH) = pywt.dwt2(gray, WAVELET)
        LLn, LHn, HLn, HHn = [x / 255. for x in (LL, LH, HL, HH)]
        both_tensor = torch.from_numpy(np.stack([LHn, HLn], 0))[None].float().to(DEVICE)
        results.append({'gray_cover': gray, 'tensor': both_tensor})
    return results

def embed_watermark(model, cover_tensor, gray_image, mapA, mapB):
    with torch.no_grad():
        z_base, _ = model(cover_tensor)
        z_emb = add_wm_split(z_base, 0, 1, mapA, mapB)
        stego_output = model(z_emb, rev=True)
        stego_coeff = stego_output[0] if isinstance(stego_output, tuple) else stego_output
        stego_img_float = coeff2img(stego_coeff, gray_image)
        stego_img_u8 = (np.clip(stego_img_float, 0, 1) * 255).round().astype(np.uint8)
        return stego_img_u8

# -------------------------- #
#  메인 실행부
# -------------------------- #
def main():
    """메인 실행 함수: 시각화 자료를 생성하고 저장합니다."""
    print(f"사용할 디바이스: {DEVICE}")
    netF = load_net(str(MODEL_FILE_PATH)) 

    image_data = load_sample_images(TEST_DIR, NUM_IMAGES_TO_SHOW)
    if not image_data:
        print("❌ ERROR: 이미지를 불러오지 못했습니다. 스크립트를 종료합니다.")
        return
    
    plot_data = []
    titles = (['Clean'] + 
              [f'JPEG {q}' for q in JPEG_Q] + 
              [f'Blur {GB_SIG[0]}'] + 
              [f'Noise {GN_SIGMA[0]}', f'Noise {GN_SIGMA[1]}', f'Noise {GN_SIGMA[2]}'])
    mapA, mapB = make_watermark()

    for sample in tqdm(image_data, desc="⚙️  스테고 이미지 생성 중"):
        gray_cover = sample['gray_cover']
        stego_clean = embed_watermark(netF, sample['tensor'], gray_cover, mapA, mapB)
        
        stego_versions = [stego_clean]
        stego_versions.extend([jpeg(stego_clean, qf) for qf in JPEG_Q])
        stego_blur_float = gblur(stego_clean.astype(np.float32)/255.0, GB_SIG[0])
        stego_versions.append((stego_blur_float * 255).round().astype(np.uint8))
        stego_versions.extend([add_noise(stego_clean, sigma) for sigma in GN_SIGMA])
            
        diff_images = []
        for stego_img in stego_versions:
            diff = np.abs(stego_img.astype(np.float32) - gray_cover.astype(np.float32))
            magnified_diff = np.clip(diff * DIFF_MAGNIFICATION, 0, 255).astype(np.uint8)
            diff_images.append(magnified_diff)
            
        plot_data.append({
            "cover": gray_cover,
            "stegos": stego_versions,
            "diffs": diff_images
        })

    # ----------------------------------------------- #
    #  ✅ 4부: 시각화 및 저장 (모든 행에 라벨 표시)
    # ----------------------------------------------- #
    print("🎨 시각화 자료를 생성합니다 (A4 크기, 간격 최소화)...")
    
    num_cols = 7 
    rows_per_image = 8 
    
    fig = plt.figure(figsize=(20, 10)) 
    
    gs = gridspec.GridSpec(
        NUM_IMAGES_TO_SHOW * rows_per_image,
        num_cols,
        figure=fig,
        width_ratios=[1] * num_cols 
    )

    for img_idx, data in enumerate(plot_data):
        base_row = img_idx * rows_per_image
        
        # --- 1. Cover Image (8줄 모두 차지) ---
        ax_cover = fig.add_subplot(gs[base_row : base_row + 8, 0])
        ax_cover.imshow(data["cover"], cmap='gray', vmin=0, vmax=255)
        ax_cover.set_ylabel(f"Cover image {img_idx+1}", fontsize=14, labelpad=15)
        ax_cover.set_xticks([])
        ax_cover.set_yticks([])

        
        # --- 2. Col 1 (Clean) & Col 2 (JPEG 90) ---
        ax_stego_c = fig.add_subplot(gs[base_row : base_row + 4, 1])
        ax_diff_c = fig.add_subplot(gs[base_row + 4 : base_row + 8, 1])
        ax_stego_c.imshow(data['stegos'][0], cmap='gray', vmin=0, vmax=255)
        ax_diff_c.imshow(data['diffs'][0], cmap='gray', vmin=0, vmax=255)
        # ✅ [수정] if img_idx == 0 제거
        ax_stego_c.set_title(titles[0], fontsize=12)
        ax_stego_c.axis('off'); ax_diff_c.axis('off')

        ax_stego_j90 = fig.add_subplot(gs[base_row : base_row + 4, 2])
        ax_diff_j90 = fig.add_subplot(gs[base_row + 4 : base_row + 8, 2])
        ax_stego_j90.imshow(data['stegos'][1], cmap='gray', vmin=0, vmax=255)
        ax_diff_j90.imshow(data['diffs'][1], cmap='gray', vmin=0, vmax=255)
        # ✅ [수정] if img_idx == 0 제거
        ax_stego_j90.set_title(titles[1], fontsize=12)
        ax_stego_j90.axis('off'); ax_diff_j90.axis('off')

        
        # --- 3. Col 3 ~ 6 (JPEG / Blur + Noise) ---
        top_row_indices = [2, 3, 4, 5] # J80, J70, J60, J50
        bot_row_indices = [6, 7, 8, 9] # Blur, N01, N02, N03
        
        for i in range(4): # 4개 컬럼 (3, 4, 5, 6)
            grid_col = i + 3 # 3번 컬럼부터 시작
            idx_top = top_row_indices[i]
            idx_bot = bot_row_indices[i]

            # 상단 블록 (JPEG)
            ax_stego_top = fig.add_subplot(gs[base_row : base_row + 2, grid_col])
            ax_diff_top = fig.add_subplot(gs[base_row + 2 : base_row + 4, grid_col])
            ax_stego_top.imshow(data['stegos'][idx_top], cmap='gray', vmin=0, vmax=255)
            ax_diff_top.imshow(data['diffs'][idx_top], cmap='gray', vmin=0, vmax=255)
            # ✅ [수정] if img_idx == 0 제거
            ax_stego_top.set_title(titles[idx_top], fontsize=12)
            ax_stego_top.axis('off'); ax_diff_top.axis('off')

            # 하단 블록 (Blur / Noise)
            ax_stego_bot = fig.add_subplot(gs[base_row + 4 : base_row + 6, grid_col])
            ax_diff_bot = fig.add_subplot(gs[base_row + 6 : base_row + 8, grid_col])
            ax_stego_bot.imshow(data['stegos'][idx_bot], cmap='gray', vmin=0, vmax=255)
            ax_diff_bot.imshow(data['diffs'][idx_bot], cmap='gray', vmin=0, vmax=255)
            # ✅ [수정] if img_idx == 0 제거
            ax_stego_bot.set_title(titles[idx_bot], fontsize=12)
            ax_stego_bot.axis('off'); ax_diff_bot.axis('off')

    plt.tight_layout(pad=0.5, h_pad=0.1, w_pad=0.5)
    plt.savefig(OUTPUT_FILENAME, bbox_inches='tight', dpi=200) 
    print(f"✨ 시각화 자료 저장이 완료되었습니다: '{OUTPUT_FILENAME}'")

if __name__ == "__main__":
    main()