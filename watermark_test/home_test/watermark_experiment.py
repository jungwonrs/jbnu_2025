일단 기존에 준 watermark_experiment 코드에서

내가 watermark embedding 부분만 좀 refactoring 해봤거든?
근데 좀 꼬인거 같어

확인좀 해줘

import os, cv2, pywt, torch, random, numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import FrEIA.framework as Ff, FrEIA.modules as Fm
from config import *
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent

# ────────── 이미지 불러오기 ──────────
def load_test_images():
    img_paths = Path(TEST_DIR)
    all_images = list(img_paths.glob("*.jpg"))
    if len(all_images) < TEST_N_IMA:
        raise ValueError(f"이미지가 {len(all_images)}장밖에 없습니다. {TEST_N_IMA}장이 필요합니다.")
    
    selected_images = random.sample(all_images, TEST_N_IMA)
    print(f"{len(selected_images)}장 이미지 선택 완료.")

    results = {}

    for test_img in selected_images:
        try:
            img = cv2.imread(str(test_img))
            img_resized = cv2.resize(img, (256, 256))
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            LL, (LH, HL, HH) = pywt.dwt2(gray, WAVELET)
            LLn, LHn, HLn, HHn = [x / 255. for x in (LL, LH, HL, HH)]

            LH_HL_tensor  = torch.from_numpy(np.stack([LHn,HLn],0))[None].float().to(DEVICE)
            LH_tensor = torch.from_numpy(np.stack([LHn,LHn],0))[None].float().to(DEVICE)
            HL_tensor = torch.from_numpy(np.stack([HLn,HLn],0))[None].float().to(DEVICE)
            FULL_tensor  = torch.from_numpy(np.stack([LLn,LHn,HLn,HHn],0))[None].float().to(DEVICE)

            results[test_img.name] = {
                'LH_HL' : LH_HL_tensor,
                'LH' : LH_tensor,
                'HL' : HL_tensor,
                'FULL' : FULL_tensor
            }
        except Exception as e:
            print(f"에러 - {test_img.name}: {e}")
    print(results)
    return results

# ────────── 모델 불러오기 ──────────
def load_net(pth):
    ck = torch.load(pth, map_location=DEVICE)
    C, H, W = ck["in_shape"]
    net = build_inn(C, H, W, ck["num_blocks"]).to(DEVICE)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net

def build_inn(C, H, W, blocks):
    nodes = [Ff.InputNode(C, H, W, name="in")]
    for k in range(blocks):
        nodes.append(Ff.Node(nodes[-1],
                             Fm.AllInOneBlock,
                             {
                                 "subnet_constructor" : subnet
                             },
                             name = f"inn_{k}"
                             ))
    nodes.append(Ff.OutputNode(
        nodes[-1],
        name = "out"
    ))
    return Ff.ReversibleGraphNet(nodes, verbose=False)

def subnet(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Conv2d(c_in, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, c_out, 3, padding=1)
    )

# ────────── 워터마크 삽입 ──────────
def embedding ():
    netA = load_net(os.path.join(MODEL_DIR, "both",  "inn_both.pth"))
    netB = load_net(os.path.join(MODEL_DIR, "lh",    "inn_lh.pth"))
    netC = load_net(os.path.join(MODEL_DIR, "hl",    "inn_hl.pth"))
    netF = load_net(os.path.join(MODEL_DIR, "full",  "inn_full.pth"))


    with torch.no_grad():
        for img_name, tensors in load_test_images():
            try:
                LH_HL = tensors['LH_HL']
                LH = tensors['LH']
                HL = tensors['HL']
                FULL = tensors['FULL']

                zA, _ = netA(LH_HL)
                stegoA, _ = netA(add_wm_split(zA, 0, 1), rev=True)

                zB, _ = netB(LH)
                stego_LH, _ = netB(add_wm_split(zB, 0, 1), rev=True)

                zC, _ = netC(HL)
                stego_HL, _ = netC(add_wm_split(zC, 0, 1), rev=True)

                LH_st = (stego_LH[0] if isinstance(stego_LH,tuple) else stego_LH)[0,0].cpu().numpy()
                HL_st = (stego_HL[0] if isinstance(stego_HL,tuple) else stego_HL)[0,0].cpu().numpy()
                stegoBC = torch.from_numpy(np.stack([LH_st,HL_st],0))[None].float().to(DEVICE)

                zF,_ = netF(FULL)
                stegoF  = netF(add_wm_split(zF,1,2), rev=True)

                recA  = coeff2img(stegoA)
                recBC = coeff2img(stegoBC)
                recF  = coeff2img(stegoF)

            except Exception as e:
                print(f"[{img_name}] 모델 처리 오류: {e}")
    return  recA, recBC, recF

@torch.no_grad()
def add_wm_split(z, chA, chB):
    rng = np.random.RandomState(WM_SEED)
    bits = rng.randint(0, 2, WM_LEN, dtype=np.uint8)
    mid  = WM_LEN // 2
    bitsA, bitsB = bits[:mid], bits[mid:]

    mapA, mapB = tile_np(bitsA), tile_np(bitsB)         
    wm_bits_rand = np.stack([mapA, mapB], 0)              

    wm_tA = torch.tensor((mapA*2-1) * WM_STRENGTH, device=DEVICE)  
    wm_tB = torch.tensor((mapB*2-1) * WM_STRENGTH, device=DEVICE)

    z2 = z.clone()
    z2[:, chA] = z2[:, chA] + wm_tA
    z2[:, chB] = z2[:, chB] + wm_tB
    return z2

def tile_np(arr):
    flat = np.repeat(arr, (128*128 + len(arr)-1)//len(arr))[:128*128]
    return flat.reshape(128, 128).astype(np.float32)

def coeff2img(coeff):
    if isinstance(coeff, tuple): coeff = coeff[0]
    if coeff.shape[1] == 2:       # LH,HL
        LL_ = pywt.dwt2(gray/255., WAVELET)[0]
        LH_,HL_ = coeff[0,0].cpu().numpy(), coeff[0,1].cpu().numpy()
        HH_ = np.zeros_like(LH_)
    else:                         # 4채널
        LL_,LH_,HL_,HH_ = [c.cpu().numpy() for c in coeff[0]]
    return pywt.idwt2((LL_,(LH_,HL_,HH_)), WAVELET)

# ────────── 워터마크 추출 ──────────
