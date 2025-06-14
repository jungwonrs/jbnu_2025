import os, cv2, pywt, torch, numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity  as ssim
import FrEIA.framework as Ff, FrEIA.modules as Fm
from config import *                       # DEVICE · WAVELET · … · MODEL_DIR
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
from model_experiment_final import make_wm_bits

# ───────── INN util ──────────────────────────────────────────
def subnet(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Conv2d(c_in, 32, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(32, c_out, 3, padding=1))

def build_inn(C, H, W, blocks):
    nodes = [Ff.InputNode(C, H, W, name="in")]
    for k in range(blocks):
        nodes.append(Ff.Node(nodes[-1], Fm.AllInOneBlock,
                             {"subnet_constructor": subnet},
                             name=f"inn_{k}"))
    nodes.append(Ff.OutputNode(nodes[-1], name="out"))
    return Ff.ReversibleGraphNet(nodes, verbose=False)

def load_net(pth):
    ck = torch.load(pth, map_location=DEVICE)
    C,H,W = ck["in_shape"]
    net = build_inn(C,H,W, ck["num_blocks"]).to(DEVICE)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net
# ────────────────────────────────────────────────────────────
netA = load_net(os.path.join(MODEL_DIR, "both",  "inn_both.pth"))
netB = load_net(os.path.join(MODEL_DIR, "lh",    "inn_lh.pth"))
netC = load_net(os.path.join(MODEL_DIR, "hl",    "inn_hl.pth"))
netF = load_net(os.path.join(MODEL_DIR, "full",  "inn_full.pth"))

# ───────── 테스트 이미지 → DWT ───────────────────────────────
test_img = ROOT_DIR / "test_image.jpg"
gray = cv2.cvtColor(cv2.resize(cv2.imread(str(test_img)), (256,256)),
                    cv2.COLOR_BGR2GRAY)
LL,(LH,HL,HH) = pywt.dwt2(gray, WAVELET)
LLn,LHn,HLn,HHn = [x/255. for x in (LL,LH,HL,HH)]

origA  = torch.from_numpy(np.stack([LHn,HLn],0))[None].float().to(DEVICE)
inp_LH = torch.from_numpy(np.stack([LHn,LHn],0))[None].float().to(DEVICE)
inp_HL = torch.from_numpy(np.stack([HLn,HLn],0))[None].float().to(DEVICE)
origF  = torch.from_numpy(np.stack([LLn,LHn,HLn,HHn],0))[None].float().to(DEVICE)

# ───────── 워터마크 두 조각(A,B) 생성 ───────────────────────
rng  = np.random.RandomState(WM_SEED)
bits = rng.randint(0, 2, WM_LEN, dtype=np.uint8)
mid  = WM_LEN // 2
bitsA, bitsB = bits[:mid], bits[mid:]

def tile_np(arr):
    flat = np.repeat(arr, (128*128 + len(arr)-1)//len(arr))[:128*128]
    return flat.reshape(128, 128).astype(np.float32)

mapA, mapB = tile_np(bitsA), tile_np(bitsB)           # (128,128)
wm_bits_rand = np.stack([mapA, mapB], 0)              # 정답 (2,128,128)

wm_tA = torch.tensor((mapA*2-1) * WM_STRENGTH, device=DEVICE)  # LH 채널용
wm_tB = torch.tensor((mapB*2-1) * WM_STRENGTH, device=DEVICE)  # HL 채널용

@torch.no_grad()
def add_wm_split(z, chA, chB):
    """LH=chA에 A, HL=chB에 B 삽입"""
    z2 = z.clone()
    z2[:, chA] = z2[:, chA] + wm_tA
    z2[:, chB] = z2[:, chB] + wm_tB
    return z2

# ───────── Stego 계수 생성 ──────────────────────────────────
with torch.no_grad():
    zA,_ = netA(origA)
    stegoA  = netA(add_wm_split(zA,0,1), rev=True)

    zB,_ = netB(inp_LH)
    ste_LH,_= netB(add_wm_split(zB,0,1), rev=True)

    zC,_ = netC(inp_HL)
    ste_HL,_= netC(add_wm_split(zC,0,1), rev=True)

    LH_st = (ste_LH[0] if isinstance(ste_LH,tuple) else ste_LH)[0,0].cpu().numpy()
    HL_st = (ste_HL[0] if isinstance(ste_HL,tuple) else ste_HL)[0,0].cpu().numpy()
    stegoBC = torch.from_numpy(np.stack([LH_st,HL_st],0))[None].float().to(DEVICE)

    zF,_ = netF(origF)
    stegoF  = netF(add_wm_split(zF,1,2), rev=True)

# ───────── 계수 → 공간영상 ───────────────────────────────────
def coeff2img(coeff):
    if isinstance(coeff, tuple): coeff = coeff[0]
    if coeff.shape[1] == 2:       # LH,HL
        LL_ = pywt.dwt2(gray/255., WAVELET)[0]
        LH_,HL_ = coeff[0,0].cpu().numpy(), coeff[0,1].cpu().numpy()
        HH_ = np.zeros_like(LH_)
    else:                         # 4채널
        LL_,LH_,HL_,HH_ = [c.cpu().numpy() for c in coeff[0]]
    return pywt.idwt2((LL_,(LH_,HL_,HH_)), WAVELET)

recA  = coeff2img(stegoA)
recBC = coeff2img(stegoBC)
recF  = coeff2img(stegoF)

# ───────── helper ─────────────────────────────────────────

def to_u8(imgf): return (np.clip(imgf,0,1)*255).round().astype(np.uint8)

JPEG_Q = [50,60,70,80,90,95]
GB_SIG = [2,3,4,5,6,7]
RS_SCALES = [0.5,0.6,0.7,0.8,0.9]       # Resize 50–90 %
CR_PCTS   = [0.6,0.7,0.8,0.9]            # Crop keep‑ratio 60–90 %
GN_SIGMA  = [0.01,0.02,0.03]             # Gaussian noise σ

# ───────── 공격 정의 ─────────────────────────────────────────
def jpeg(u8,q):
    enc = cv2.imencode('.jpg',u8,[cv2.IMWRITE_JPEG_QUALITY,q])[1]
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

def gblur(f,s):
    k = int(s*4+1)|1
    return cv2.GaussianBlur(f,(k,k),sigmaX=s,borderType=cv2.BORDER_REPLICATE)

def resize_scale(u8, scale):
    small = cv2.resize(u8, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, u8.shape[::-1], interpolation=cv2.INTER_LINEAR)

def crop_pct(u8, keep):
    h,w = u8.shape
    ph,pw = int(h*keep), int(w*keep)
    y0,x0 = (h-ph)//2, (w-pw)//2
    crop = u8[y0:y0+ph, x0:x0+pw]
    return cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)

def add_noise(u8, sigma):
    f = u8.astype(np.float32)/255.
    noisy = np.clip(f + np.random.randn(*f.shape)*sigma, 0,1)
    return (noisy*255).round().astype(np.uint8)

# ───────── spatial→coeff ────────────────────────────
def spatial2coeff(img_f, mode):
    LL,(LH,HL,HH)=pywt.dwt2(img_f*255., WAVELET)
    LL,LH,HL,HH=[x/255. for x in (LL,LH,HL,HH)]
    return torch.from_numpy(
        np.stack([LH,HL],0) if mode=="2" else np.stack([LL,LH,HL,HH],0)
    )[None].float().to(DEVICE)

# ───────── 워터마크 추출 & 평가 ──────────────────────────────
@torch.no_grad()
def extract(coeff: torch.Tensor,
            model: torch.nn.Module,
            two_ch: bool,
            z_base: torch.Tensor) -> np.ndarray:
    z, _ = model(coeff)
    if not two_ch:            # Full → LH·HL 만 비교
        z, z_base = z[:, 1:3], z_base[:, 1:3]

    logits = (z - z_base) * (SCALE_LOGIT / WM_STRENGTH)  # (B,2,H,W)

    # ✔ 128×128 전 영역 평균 → **공격 노이즈 평균화**
    bit_logits = logits.mean(dim=(-2, -1))               # (B,2)
    pred_bits = (bit_logits > 0).to(torch.uint8)         # {0,1}

    return pred_bits[0].detach().cpu().numpy()                    # shape (2,)

@torch.no_grad()
def extract_two_nets(coeff: torch.Tensor,
                     net_lh, net_hl,
                     z_base_lh, z_base_hl) -> np.ndarray:
    """LH 비트는 net_lh,  HL 비트는 net_hl 에서 각각 복원"""
    # LH 계수만 2-채널로 복제 → net_lh 입력
    LH = coeff[:, 0:1]                       # (1,1,H,W)
    coeff_lh = torch.cat([LH, LH], 1)
    bit_lh = extract(coeff_lh, net_lh, True, z_base_lh)[0]

    # HL 계수도 동일
    HL = coeff[:, 1:2]
    coeff_hl = torch.cat([HL, HL], 1)
    bit_hl = extract(coeff_hl, net_hl, True, z_base_hl)[0]

    return np.array([bit_lh, bit_hl], dtype=np.uint8)

def wm_metrics(pred: np.ndarray, gt: np.ndarray):
    if gt.ndim == 3:
        gt = (gt.mean(axis=(1, 2)) > 0.5).astype(np.uint8)

    # 정확도 & BER
    acc = (pred == gt).mean()
    ber = (pred != gt).mean()

    # NC 계산 - type 안전성 확보
    pred = pred.astype(np.int8)
    gt   = gt.astype(np.int8)
    pred_bipolar = pred * 2 - 1
    gt_bipolar   = gt * 2 - 1
    nc = (pred_bipolar * gt_bipolar).mean()

    return acc, ber, nc

def if_rmse(rec): return float(np.sqrt(np.mean((rec - gray/255.)**2)))


# ───────── 평가 함수 (단일 INN) ──────────────────────────────
def evaluate(tag, rec_img, coeff_tensor, model, two_ch, z_base):
    base_u8 = to_u8(rec_img)
    p0,s0 = psnr(gray, base_u8, data_range=255), ssim(gray, base_u8, data_range=255, win_size=7)
    def _log(msg): print(f"[{tag:5s} | {msg}")

    # clean
    pred = extract(coeff_tensor, model, two_ch, z_base)
    acc,ber,nc = wm_metrics(pred, wm_bits_rand)
    _log(f"clean ]  PSNR {p0:6.2f} SSIM {s0:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

    # JPEG
    for q in JPEG_Q:
        atk_u8 = jpeg(base_u8,q); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2" if two_ch else "4")
        acc,ber,nc = wm_metrics(extract(atk_coef, model, two_ch, z_base), wm_bits_rand)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"JPEG{q:02d}] PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

    # Gaussian Blur
    for sg in GB_SIG:
        atk_f = gblur(rec_img, sg); atk_coef = spatial2coeff(atk_f, "2" if two_ch else "4")
        acc,ber,nc = wm_metrics(extract(atk_coef, model, two_ch, z_base), wm_bits_rand)
        atk_u8 = to_u8(atk_f); p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Blur{sg}]  PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

    # Resize
    for sc in RS_SCALES:
        atk_u8 = resize_scale(base_u8, sc); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2" if two_ch else "4")
        acc,ber,nc = wm_metrics(extract(atk_coef, model, two_ch, z_base), wm_bits_rand)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Res{int(sc*100):02d}]  PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

    # Center Crop
    for keep in CR_PCTS:
        atk_u8 = crop_pct(base_u8, keep); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2" if two_ch else "4")
        acc,ber,nc = wm_metrics(extract(atk_coef, model, two_ch, z_base), wm_bits_rand)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Crop{int(keep*100):02d}] PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

    # Gaussian Noise
    for sg in GN_SIGMA:
        atk_u8 = add_noise(base_u8, sg); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2" if two_ch else "4")
        acc,ber,nc = wm_metrics(extract(atk_coef, model, two_ch, z_base), wm_bits_rand)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Noise{int(sg*100):02d}] PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

# ───────── B+C 전용 평가 함수 (다중 INN)──────────────────────────────
def evaluate_BC(tag, rec_img, coeff_tensor, net_lh, net_hl, z_base_lh, z_base_hl):
    base_u8 = to_u8(rec_img)
    p0,s0 = psnr(gray, base_u8, data_range=255), ssim(gray, base_u8, data_range=255, win_size=7)
    def _log(msg): print(f"[{tag:5s} | {msg}")

    pred = extract_two_nets(coeff_tensor, net_lh, net_hl, z_base_lh, z_base_hl)
    acc,ber,nc = wm_metrics(pred, wm_bits_rand)
    _log(f"clean ]  PSNR {p0:6.2f} SSIM {s0:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

    # JPEG
    for q in JPEG_Q:
        atk_u8 = jpeg(base_u8,q); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef  = spatial2coeff(atk_f,"2")
        acc,ber,nc = wm_metrics(extract_two_nets(atk_coef, net_lh, net_hl, z_base_lh, z_base_hl), wm_bits_rand)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"JPEG{q:02d}] PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

    # Gaussian Blur
    for sg in GB_SIG:
        atk_f = gblur(rec_img, sg); atk_coef = spatial2coeff(atk_f,"2")
        acc,ber,nc = wm_metrics(extract_two_nets(atk_coef, net_lh, net_hl, z_base_lh, z_base_hl), wm_bits_rand)
        atk_u8 = to_u8(atk_f); p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Blur{sg}]  PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

    # Resize
    for sc in RS_SCALES:
        atk_u8 = resize_scale(base_u8, sc); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2")
        acc,ber,nc = wm_metrics(extract_two_nets(atk_coef, net_lh, net_hl, z_base_lh, z_base_hl), wm_bits_rand)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Res{int(sc*100):02d}]  PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

    # Center Crop
    for keep in CR_PCTS:
        atk_u8 = crop_pct(base_u8, keep); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2")
        acc,ber,nc = wm_metrics(extract_two_nets(atk_coef, net_lh, net_hl, z_base_lh, z_base_hl), wm_bits_rand)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Crop{int(keep*100):02d}] PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")

    # Gaussian Noise
    for sg in GN_SIGMA:
        atk_u8 = add_noise(base_u8, sg); atk_f = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2")
        acc,ber,nc = wm_metrics(extract_two_nets(atk_coef, net_lh, net_hl, z_base_lh, z_base_hl), wm_bits_rand)
        p,s = psnr(gray, atk_u8, data_range=255), ssim(gray, atk_u8, data_range=255, win_size=7)
        _log(f"Noise{int(sg*100):02d}] PSNR {p:6.2f} SSIM {s:.4f} | ACC {acc*100:6.2f}% BER {ber*100:5.2f}% NC {nc:.3f}")


with torch.no_grad():
    z_clean_A,_ = netA(origA)
    z_clean_B,_ = netB(inp_LH)
    z_clean_C,_ = netC(inp_HL)
    z_clean_F,_ = netF(origF)

print("\n### Robustness evaluation ###\n")

evaluate("A-only", recA,
         stegoA if not isinstance(stegoA, tuple) else stegoA[0],
         netA, True, z_clean_A)

evaluate_BC("B+C", recBC, stegoBC,
            netB, netC,
            z_clean_B, z_clean_C)

evaluate("Full", recF,
         stegoF if not isinstance(stegoF, tuple) else stegoF[0],
         netF, False, z_clean_F)
