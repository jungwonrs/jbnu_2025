# ───────── watermark_experiment_final2.py ─────────
"""
● Stego 생성 → (clean · JPEG50~95 · GaussianBlur σ2~5) 강인성 평가
  - PSNR / SSIM
  - ACC / BER / NC / IF-RMSE
  - 결과는 run2.py 가 STDOUT 을 캡처해 common-log 로 모읍니다.
"""

import os, cv2, pywt, torch, numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity  as ssim
import FrEIA.framework as Ff, FrEIA.modules as Fm
from config import *                       # DEVICE · WAVELET · … · MODEL_DIR
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent

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

# ───────── 워터마크 (±1 평균 → scalar) ───────────────────────
bits = ''.join(f'{ord(c):08b}' for c in WM_STRING)
vec  = (np.frombuffer(bits.encode(),np.uint8)-48)*2-1        # (+1/-1)
wm_scalar = float(vec.mean()) * WM_STRENGTH                  # 하나의 스칼라
half1, half2 = wm_scalar*0.5, wm_scalar*0.5                  # LH / HL

@torch.no_grad()
def add_wm(z, chL, chH, h1, h2):
    z2 = z.clone();  z2[:,chL] += h1;  z2[:,chH] += h2;  return z2

# ───────── Stego 계수 생성 (깨끗한 상태) ─────────────────────
with torch.no_grad():
    zA,_ = netA(origA);  stegoA  = netA(add_wm(zA,0,1,half1,half2), rev=True)

    zB,_ = netB(inp_LH); ste_LH,_= netB(add_wm(zB,0,1,half1,0.0),   rev=True)
    zC,_ = netC(inp_HL); ste_HL,_= netC(add_wm(zC,0,1,0.0,  half2), rev=True)
    LH_st = (ste_LH[0] if isinstance(ste_LH,tuple) else ste_LH)[0,0].cpu().numpy()
    HL_st = (ste_HL[0] if isinstance(ste_HL,tuple) else ste_HL)[0,0].cpu().numpy()
    stegoBC = torch.from_numpy(np.stack([LH_st,HL_st],0))[None].float().to(DEVICE)

    zF,_ = netF(origF);  stegoF  = netF(add_wm(zF,1,2,half1,half2), rev=True)

# ───────── 계수 → 공간영상 ───────────────────────────────────
def coeff2img(coeff):
    if isinstance(coeff, tuple): coeff = coeff[0]
    C = coeff.shape[1]
    if C==2:           # LH,HL 만
        LL_ = pywt.dwt2(gray/255., WAVELET)[0]
        LH_,HL_ = coeff[0,0].cpu().numpy(), coeff[0,1].cpu().numpy()
        HH_ = np.zeros_like(LH_)
    else:               # LL,LH,HL,HH
        LL_,LH_,HL_,HH_ = [c.cpu().numpy() for c in coeff[0]]
    return pywt.idwt2((LL_,(LH_,HL_,HH_)), WAVELET)

recA  = coeff2img(stegoA)
recBC = coeff2img(stegoBC)
recF  = coeff2img(stegoF)

def to_u8(imgf): return (np.clip(imgf,0,1)*255).round().astype(np.uint8)

# ───────── 공격 정의 ─────────────────────────────────────────
JPEG_Q = [50,60,70,80,90,95]
GB_SIG = [2,3,4,5]

def jpeg(u8,q):
    enc = cv2.imencode('.jpg',u8,[cv2.IMWRITE_JPEG_QUALITY,q])[1]
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

def gblur(f,s):
    k = int(s*4+1)|1
    return cv2.GaussianBlur(f,(k,k),sigmaX=s,borderType=cv2.BORDER_REPLICATE)

# ───────── helper: spatial→coeff (2ch 또는 4ch) ───────────────
def spatial2coeff(img_f, mode):
    LL,(LH,HL,HH)=pywt.dwt2(img_f*255., WAVELET)
    LL,LH,HL,HH=[x/255. for x in (LL,LH,HL,HH)]
    if mode=="2":
        arr=np.stack([LH,HL],0)
    else:
        arr=np.stack([LL,LH,HL,HH],0)
    return torch.from_numpy(arr)[None].float().to(DEVICE)

# ───────── 워터마크 추출 (±WM_STRENGTH) ───────────────────────
wm_bits_rand = np.random.randint(0,2,(2,128,128)).astype(np.float32)
wm_t_rand    = torch.tensor((wm_bits_rand*2-1)*WM_STRENGTH,device=DEVICE)

@torch.no_grad()
def extract(coeff, model, two_ch):
    z,_ = model(coeff)
    if not two_ch: z = z[:,1:3]     # Full ⇒ LH,HL 부분만
    return (z[0].cpu().numpy()>0).astype(np.uint8)

def wm_metrics(pred, gt: np.ndarray):
    pred_b = (pred > 0).astype(np.uint8)  # Bool → {0,1}
    gt_b   = (gt   > 0).astype(np.uint8)

    ber = (pred_b != gt_b).mean()
    acc = 1.0 - ber
    nc  = np.logical_and(pred_b, gt_b).sum() / gt_b.size
    return acc, ber, nc

def if_rmse(rec): return float(np.sqrt(np.mean((rec - gray/255.)**2)))

# ───────── 평가 함수 ──────────────────────────────────────────
def evaluate(tag, rec_img, coeff_tensor, model, two_ch):
    base_u8 = to_u8(rec_img)
    p0 = psnr(gray, base_u8, data_range=255)
    s0 = ssim(gray, base_u8, data_range=255, win_size=7)
    wm_pred = extract(coeff_tensor, model, two_ch)
    acc,ber,nc = wm_metrics(wm_pred, wm_bits_rand)
    print(f"[{tag:5s} | clean ]  PSNR {p0:6.2f}  SSIM {s0:.4f} | "
          f"ACC {acc*100:6.2f}%  BER {ber*100:5.2f}%  NC {nc:.3f}  IF {if_rmse(rec_img):.5f}")

    # JPEG
    for q in JPEG_Q:
        atk_u8 = jpeg(base_u8,q)
        atk_f  = atk_u8.astype(np.float32)/255.
        atk_coef = spatial2coeff(atk_f, "2" if two_ch else "4")
        wm_pred = extract(atk_coef, model, two_ch)
        acc,ber,nc = wm_metrics(wm_pred, wm_bits_rand)
        p = psnr(gray, atk_u8, data_range=255)
        s = ssim(gray, atk_u8, data_range=255, win_size=7)
        print(f"[{tag:5s} | JPEG{q:02d}] PSNR {p:6.2f}  SSIM {s:.4f} | "
              f"ACC {acc*100:6.2f}%  BER {ber*100:5.2f}%  NC {nc:.3f}  IF {if_rmse(atk_f):.5f}")
    # Gaussian Blur
    for sg in GB_SIG:
        atk_f = gblur(rec_img, sg)
        atk_coef = spatial2coeff(atk_f, "2" if two_ch else "4")
        wm_pred = extract(atk_coef, model, two_ch)
        acc,ber,nc = wm_metrics(wm_pred, wm_bits_rand)
        atk_u8 = to_u8(atk_f)
        p = psnr(gray, atk_u8, data_range=255)
        s = ssim(gray, atk_u8, data_range=255, win_size=7)
        print(f"[{tag:5s} | GaussianBlur{sg}]  PSNR {p:6.2f}  SSIM {s:.4f} | "
              f"ACC {acc*100:6.2f}%  BER {ber*100:5.2f}%  NC {nc:.3f}  IF {if_rmse(atk_f):.5f}")

print("\n### Robustness evaluation (JPEG / GaussianBlur) ###\n")
evaluate("A-only", recA,  stegoA if not isinstance(stegoA,tuple) else stegoA[0], netA, True)
evaluate("B+C  ", recBC, stegoBC, netA, True)   # 2-채널 모델(netA)로 추출
evaluate("Full ", recF,  stegoF if not isinstance(stegoF,tuple) else stegoF[0], netF, False)
# ─────────────────────────────────────────────────────────────
