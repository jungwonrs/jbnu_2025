#
# watermark_experiment.py
#
import os, cv2, pywt, torch, random, numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import FrEIA.framework as Ff, FrEIA.modules as Fm
from pathlib import Path
from config import *
from collections import defaultdict
import logging, sys, os
import torch.nn.functional as F


log_fname = os.environ.get("LOG_FILE", "results.txt") 

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_fname, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parent

# ────────── 이미지 불러오기 ──────────
def load_test_images():
    img_paths = Path(TEST_DIR)
    all_images = list(img_paths.glob("*.jpg"))
    if len(all_images) < TEST_N_IMG:
        raise ValueError(f"이미지가 {len(all_images)}장밖에 없습니다. {TEST_N_IMG}장이 필요합니다.")
    
    selected_images = random.sample(all_images, TEST_N_IMG)
    #print(f"{len(selected_images)}장 이미지 선택 완료.")

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
                'gray' : gray,
                'LH_HL' : LH_HL_tensor,
                'LH' : LH_tensor,
                'HL' : HL_tensor,
                'FULL' : FULL_tensor
            }
        except Exception as e:
            logger.info(f"에러 - {test_img.name}: {e}")
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

#
# [수정] 모든 모델을 로드하도록 변경
#
def load_all_nets():
    netA = load_net(os.path.join(MODEL_DIR, "both",  "inn_both.pth"))
    netB = load_net(os.path.join(MODEL_DIR, "lh",    "inn_lh.pth"))
    netC = load_net(os.path.join(MODEL_DIR, "hl",    "inn_hl.pth"))
    netF = load_net(os.path.join(MODEL_DIR, "full",  "inn_full.pth"))

    return netA, netB, netC, netF

# ────────── 워터마크 삽입 ──────────
def make_watermark():
    rng = np.random.RandomState(WM_SEED)
    bits = rng.randint(0, 2, WM_LEN, dtype=np.uint8)
    mid = WM_LEN // 2
    bitsA, bitsB = bits[:mid], bits[mid:]
    mapA, mapB = tile_np(bitsA), tile_np(bitsB)
    wm_bits_rand = np.stack([mapA, mapB], 0)
    return bitsA, bitsB, mapA, mapB, wm_bits_rand

#
# [수정] 모든 모델을 사용하여 워터마크를 삽입하도록 변경
#
def embedding (netA, netB, netC, netF):
    results = {}

    with torch.no_grad():
        for img_name, tensors in load_test_images().items():
            try:
                gray = tensors['gray']
                LH_HL = tensors['LH_HL']
                LH = tensors['LH']
                HL = tensors['HL']
                FULL = tensors['FULL']
                bitsA, bitsB, mapA, mapB, wm_bits_rand = make_watermark()

                zA, _ = netA(LH_HL)
                stegoA = netA(add_wm_split(zA, 0, 1, mapA, mapB), rev=True)

                zB, _ = netB(LH)
                stego_LH = netB(add_wm_split(zB, 0, 0, mapA, mapA), rev=True)

                zC, _ = netC(HL)
                stego_HL = netC(add_wm_split(zC, 0, 0, mapB, mapB), rev=True)

                LH_st = (stego_LH[0] if isinstance(stego_LH,tuple) else stego_LH)[0,0].cpu().numpy()
                HL_st = (stego_HL[0] if isinstance(stego_HL,tuple) else stego_HL)[0,0].cpu().numpy()
                stegoBC = torch.from_numpy(np.stack([LH_st,HL_st],0))[None].float().to(DEVICE)

                zF,_ = netF(FULL)
                stegoF = netF(add_wm_split(zF, 1, 2,  mapA, mapB), rev=True)

                recA  = coeff2img(stegoA, gray)
                recBC = coeff2img(stegoBC, gray)
                recF  = coeff2img(stegoF, gray)

                results[img_name] = {
                    'coeffA_clean': LH_HL,
                    'coeffB_clean': LH,
                    'coeffC_clean': HL,
                    'coeffF_clean': FULL,
                    'coeffA_stego': stegoA[0] if isinstance(stegoA, tuple) else stegoA,
                    'coeffBC_stego': stegoBC,
                    'coeffF_stego': stegoF[0] if isinstance(stegoF, tuple) else stegoF,

                    'recA' : recA,
                    'recBC': recBC,
                    'recF': recF,
                    
                    'gray': gray,

                    'wm_bits_rand': wm_bits_rand,
                    'bitsA': bitsA,
                    'bitsB': bitsB,
                    'gt_bits'     : np.concatenate([bitsA, bitsB]),
                }

            except Exception as e:
                logger.info(f"[{img_name}] 모델 처리 오류: {e}")
    return  results

@torch.no_grad()
def add_wm_split(z, chA, chB, mapA, mapB):
    wm_tA = torch.tensor((mapA*2-1) * WM_STRENGTH, device=DEVICE)  
    wm_tB = torch.tensor((mapB*2-1) * WM_STRENGTH, device=DEVICE)
    z2 = z.clone()
    z2[:, chA] = z2[:, chA] + wm_tA
    z2[:, chB] = z2[:, chB] + wm_tB
    return z2

def tile_np(arr):
    flat = np.repeat(arr, (128*128 + len(arr)-1)//len(arr))[:128*128]
    return flat.reshape(128, 128).astype(np.float32)

def coeff2img(coeff, gray):
    if isinstance(coeff, tuple): coeff = coeff[0]
    if coeff.shape[1] == 2:       # LH,HL
        LL_ = pywt.dwt2(gray/255., WAVELET)[0]
        LH_,HL_ = coeff[0,0].cpu().numpy(), coeff[0,1].cpu().numpy()
        HH_ = np.zeros_like(LH_)
    else:                         # 4채널
        LL_,LH_,HL_,HH_ = [c.cpu().numpy() for c in coeff[0]]
    return pywt.idwt2((LL_,(LH_,HL_,HH_)), WAVELET)

# ───────── 수정된 extract ─────────
@torch.no_grad()
def extract(coeff, model, two_ch, z_base):
    z, _ = model(coeff)
    if not two_ch:                       # full INN → LH·HL 두 채널만 사용
        z, z_base = z[:, 1:3], z_base[:, 1:3]   # (B,2,H,W)

    dz = z - z_base
    # 채널·공간 평균을 한 번 더 제거
    mu = dz.flatten(2).mean(-1, keepdim=True).unsqueeze(-1)            # (B,C,1,1)
    s  = dz.flatten(2).std(-1, keepdim=True).unsqueeze(-1) + 1e-6      # (B,C,1,1)
    logits = ((dz - mu) / s) * LOGIT_COEFF
    B, C, H, W = logits.shape            # C = 2

    # ── 채널별로 128 비트씩 풀링 ──
    bit_idx = (torch.arange(H*W, device=logits.device)
               * (WM_LEN // 2) // (H*W)).view(1,1,H,W)
    pooled = torch.zeros(B, C, WM_LEN//2, device=logits.device).scatter_add_(
        2,
        bit_idx.expand(B,C,H,W).reshape(B, C, -1),
        logits.reshape(B, C, -1)
    ) / (H * W / (WM_LEN//2))

    bits_lh = (pooled[:,0] > 0).to(torch.uint8)      # LH → 앞 128
    bits_hl = (pooled[:,1] > 0).to(torch.uint8)      # HL → 뒤 128
    pred_bits = torch.cat([bits_lh, bits_hl], dim=1) # (B,256)
    return pred_bits.squeeze(0).cpu().numpy()


# ───────── 수정된 extract_two_nets ─────────
@torch.no_grad()
def extract_two_nets(coeff, net_lh, net_hl, z_base_lh, z_base_hl):
    # LH
    bits_lh = extract(coeff[:, 0:1].repeat(1,2,1,1), net_lh, True, z_base_lh)  # (256,)
    # HL
    bits_hl = extract(coeff[:, 1:2].repeat(1,2,1,1), net_hl, True, z_base_hl)

    pred_bits = np.concatenate([bits_lh[:128], bits_hl[:128]]).astype(np.uint8)
    return pred_bits


# ───────── 완전히 교체할 wm_metrics ─────────
def wm_metrics(pred, gt):
    """
    pred / gt 가 (256,) 이든 (2,128,128)이든
    길이 256의 1-차원 벡터로 맞춘 뒤
    ACC, BER, NC를 계산한다.
    """
    # 1) 1-차원으로 평탄화 + uint8 변환
    pred = pred.reshape(-1).astype(np.int16)
    gt   = gt.reshape(-1).astype(np.int16)

    # 2) 길이 확인
    assert pred.shape == gt.shape, "shape mismatch!"

    # 3) ACC / BER
    acc = (pred == gt).mean()
    ber = 1.0 - acc

    # 4) NC  (0/1 → -1/+1 후 코사인 유사도)
    pred_bipolar = (pred * 2 - 1).astype(np.float32)
    gt_bipolar   = (gt   * 2 - 1).astype(np.float32)
    nc = (pred_bipolar * gt_bipolar).mean()

    return acc, ber, nc

def evaluate(tag, rec_img, coeff_tensor, model, two_ch, z_base, gray, gt_bits):
    base_u8 = to_u8(rec_img)
    p0,s0 = psnr(gray, base_u8, data_range=255), ssim(gray, base_u8, data_range=255, win_size=7)
    def _log(msg): logger.info(f"[{tag:5s}] {msg}")

    # clean
    coeff_from_spatial = spatial2coeff(rec_img, "2" if two_ch else "4")
    pred = extract(coeff_from_spatial, model, two_ch, z_base)

    acc0,ber0,nc0 = wm_metrics(pred, gt_bits)
    _log(f"clean ]  PSNR {p0:6.2f} SSIM {s0:.4f} | ACC {acc0*100:6.2f}% BER {ber0*100:5.2f}% NC {nc0:.3f}")

    metrics = {"clean": {"psnr": p0, "ssim": s0, "acc": acc0, "ber": ber0, "nc": nc0}}

    # JPEG
    for q in JPEG_Q:
        atk_u8 = jpeg(base_u8,q); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2" if two_ch else "4")
        acc,ber,nc = wm_metrics(extract(atk_coef, model, two_ch, z_base), gt_bits)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"JPEG{q:02d}] PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

        metrics[f"jpeg{q}"] = {"psnr": p, "ssim": s, "acc": acc, "ber": ber, "nc": nc}


    # Gaussian Blur
    for sg in GB_SIG:
        atk_f = gblur(rec_img, sg); atk_coef = spatial2coeff(atk_f, "2" if two_ch else "4")
        acc,ber,nc = wm_metrics(extract(atk_coef, model, two_ch, z_base), gt_bits)
        atk_u8 = to_u8(atk_f); p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Blur{sg}]  PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

        metrics[f"blur{sg}"] = {"psnr": p, "ssim": s, "acc": acc, "ber": ber, "nc": nc}

    '''
    # Resize
    for sc in RS_SCALES:
        atk_u8 = resize_scale(base_u8, sc); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2" if two_ch else "4")
        acc,ber,nc = wm_metrics(extract(atk_coef, model, two_ch, z_base), gt_bits)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Res{int(sc*100):02d}]  PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

        metrics[f"res{int(sc*100)}"] = {"psnr": p, "ssim": s, "acc": acc, "ber": ber, "nc": nc}

    # Center Crop
    for keep in CR_PCTS:
        atk_u8 = crop_pct(base_u8, keep); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2" if two_ch else "4")
        acc,ber,nc = wm_metrics(extract(atk_coef, model, two_ch, z_base), gt_bits)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Crop{int(keep*100):02d}] PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

        metrics[f"crop{int(keep*100)}"] = {"psnr": p, "ssim": s, "acc": acc, "ber": ber, "nc": nc}
    '''

    # Gaussian Noise
    for sg in GN_SIGMA:
        atk_u8 = add_noise(base_u8, sg); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2" if two_ch else "4")
        acc,ber,nc = wm_metrics(extract(atk_coef, model, two_ch, z_base), gt_bits)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Noise{int(sg*100):02d}] PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

        metrics[f"noise{int(sg*100)}"] = {"psnr": p, "ssim": s, "acc": acc, "ber": ber, "nc": nc}

    
    return metrics

def evaluate_BC(tag, rec_img, coeff_tensor, net_lh, net_hl, z_base_lh, z_base_hl, gray, gt_bits):
    base_u8 = to_u8(rec_img)
    p0,s0 = psnr(gray, base_u8, data_range=255), ssim(gray, base_u8, data_range=255, win_size=7)
    def _log(msg): logger.info(f"[{tag:5s}] {msg}")

    #pred = extract_two_nets(coeff_tensor, net_lh, net_hl, z_base_lh, z_base_hl)
    coeff_from_spatial = spatial2coeff(rec_img, "2")
    pred = extract_two_nets(coeff_from_spatial, net_lh, net_hl, z_base_lh, z_base_hl)
    acc0,ber0,nc0 = wm_metrics(pred, gt_bits)
    _log(f"clean ]  PSNR {p0:6.2f} SSIM {s0:.4f} | ACC {acc0*100:6.2f}% BER {ber0*100:5.2f}% NC {nc0:.3f}")

    metrics = {"clean": {"psnr": p0, "ssim": s0, "acc": acc0, "ber": ber0,  "nc": nc0}}


    # JPEG
    for q in JPEG_Q:
        atk_u8 = jpeg(base_u8,q); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef  = spatial2coeff(atk_f,"2")
        acc,ber,nc = wm_metrics(extract_two_nets(atk_coef, net_lh, net_hl, z_base_lh, z_base_hl), gt_bits)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"JPEG{q:02d}] PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")
        metrics[f"jpeg{q}"] = {"psnr": p, "ssim": s, "acc": acc, "ber": ber, "nc": nc}

    # Gaussian Blur
    for sg in GB_SIG:
        atk_f = gblur(rec_img, sg); atk_coef = spatial2coeff(atk_f,"2")
        acc,ber,nc = wm_metrics(extract_two_nets(atk_coef, net_lh, net_hl, z_base_lh, z_base_hl), gt_bits)
        atk_u8 = to_u8(atk_f); p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Blur{sg}]  PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")
        metrics[f"blur{sg}"] = {"psnr": p, "ssim": s, "acc": acc, "ber": ber, "nc": nc}
    '''
    # Resize
    for sc in RS_SCALES:
        atk_u8 = resize_scale(base_u8, sc); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2")
        acc,ber,nc = wm_metrics(extract_two_nets(atk_coef, net_lh, net_hl, z_base_lh, z_base_hl), gt_bits)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Res{int(sc*100):02d}]  PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")
        metrics[f"res{int(sc*100)}"] = {"psnr": p, "ssim": s, "acc": acc, "ber": ber, "nc": nc}

    # Center Crop
    for keep in CR_PCTS:
        atk_u8 = crop_pct(base_u8, keep); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2")
        acc,ber,nc = wm_metrics(extract_two_nets(atk_coef, net_lh, net_hl, z_base_lh, z_base_hl), gt_bits)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Crop{int(keep*100):02d}] PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")
        metrics[f"crop{int(keep*100)}"] = {"psnr": p, "ssim": s, "acc": acc, "ber": ber, "nc": nc}
    '''
    # Gaussian Noise
    for sg in GN_SIGMA:
        atk_u8 = add_noise(base_u8, sg); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2")
        acc,ber,nc = wm_metrics(extract_two_nets(atk_coef, net_lh, net_hl, z_base_lh, z_base_hl), gt_bits)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Noise{int(sg*100):02d}] PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")
        metrics[f"noise{int(sg*100)}"] = {"psnr": p, "ssim": s, "acc": acc, "ber": ber, "nc": nc}

    return metrics



# ────────── 유틸 ──────────
def to_u8(imgf): 
    return (np.clip(imgf,0,1)*255).round().astype(np.uint8)

def jpeg(u8,q):
    enc = cv2.imencode('.jpg',u8,[cv2.IMWRITE_JPEG_QUALITY,q])[1]
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

def gblur(f,s):
    k = int(s*4+1)|1
    return cv2.GaussianBlur(f,(k,k),sigmaX=s,borderType=cv2.BORDER_REPLICATE)

'''
def resize_scale(u8, scale):
    small = cv2.resize(u8, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, u8.shape[::-1], interpolation=cv2.INTER_LINEAR)

def crop_pct(u8, keep):
    h,w = u8.shape
    ph,pw = int(h*keep), int(w*keep)
    y0,x0 = (h-ph)//2, (w-pw)//2
    crop = u8[y0:y0+ph, x0:x0+pw]
    return cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)
'''
def add_noise(u8, sigma):
    f = u8.astype(np.float32)/255.
    noisy = np.clip(f + np.random.randn(*f.shape)*sigma, 0,1)
    return (noisy*255).round().astype(np.uint8)

def spatial2coeff(img_f, mode):
    LL,(LH,HL,HH)=pywt.dwt2(img_f*255., WAVELET)
    LL,LH,HL,HH=[x/255. for x in (LL,LH,HL,HH)]
    return torch.from_numpy(
        np.stack([LH,HL],0) if mode=="2" else np.stack([LL,LH,HL,HH],0)
    )[None].float().to(DEVICE)

#
# [수정] 모든 모델을 로드하고 평가하도록 실행부 변경
#
# ────────── 실행 ──────────
netA, netB, netC, netF = load_all_nets()
results = embedding(netA, netB, netC, netF)

sum_metrics = defaultdict(lambda: defaultdict(lambda: {'psnr': 0.0,
                                                       'ssim': 0.0,
                                                       'acc': 0.0,
                                                       'ber':  0.0,
                                                       'nc': 0.0}))
cnt_metrics = defaultdict(lambda: defaultdict(int))

logger.info("\n### Robustness evaluation ###")
for img_name, D in results.items():
    logger.info(f"\n===== {img_name} =====")

    # ─ clean 계수 → z ─
    zA,_ = netA(D["coeffA_clean"])
    zB,_ = netB(D["coeffB_clean"])
    zC,_ = netC(D["coeffC_clean"])
    zF,_ = netF(D["coeffF_clean"])

    # ─ (1) A-only ─
    mA = evaluate("A-only",
                  D["recA"], D["coeffA_stego"],
                  netA, True, zA,
                  D["gray"], D["gt_bits"])

    # ─ (2) B+C ─
    mBC = evaluate_BC("B+C",
                      D["recBC"], D["coeffBC_stego"],
                      netB, netC, zB, zC,
                      D["gray"], D["gt_bits"])

    # ─ (3) Full ─
    mF = evaluate("Full",
                  D["recF"], D["coeffF_stego"],
                  netF, False, zF,
                  D["gray"], D["gt_bits"])

    # ─── 합계/카운트 누적 ───
    for tag, m in [('A-only', mA), ('B+C', mBC), ('Full', mF)]:
        for atk, vals in m.items(): 
            for k in ('psnr', 'ssim', 'acc', 'ber', 'nc'):
                sum_metrics[tag][atk][k] += vals[k]
            cnt_metrics[tag][atk] += 1

# ───────── 평균 출력 ─────────
logger.info("\n===== AVERAGE OVER ALL IMAGES =====")
header = "{:<8} {:<8} {:>9} {:>11} {:>10} {:>10} {:>8}"
logger.info(header.format("Tag", "Attack", 
                          "Avg-PSNR", "Avg-SSIM", "Avg-acc", "Avg-BER", "Avg-nc"))

attack_order = (
    ['clean'] +
    [f'jpeg{q}' for q in JPEG_Q] +
    [f'blur{sg}' for sg in GB_SIG] +
    #[f'res{int(s*100)}' for s in RS_SCALES] +
    #[f'crop{int(c*100)}' for c in CR_PCTS] +
    [f'noise{int(n*100)}' for n in GN_SIGMA]
)

for tag in ["A-only", "B+C", "Full"]:
    for atk in attack_order:
        if cnt_metrics[tag][atk] == 0:
            continue
        n  = cnt_metrics[tag][atk]
        ps = sum_metrics[tag][atk]['psnr'] / n
        ss = sum_metrics[tag][atk]['ssim'] / n
        ac = sum_metrics[tag][atk]['acc'] / n
        be = sum_metrics[tag][atk]['ber']  / n
        nc = sum_metrics[tag][atk]['nc']   / n
        logger.info(header.format(tag, atk, f"{ps:.2f}", f"{ss:.4f}", f"{ac:.4f}",  f"{be:.4f}", f"{nc:.3f}"))