# ───────── watermark_experiment_full.py ──────────
import cv2, pywt, torch, numpy as np, matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity  as ssim
import FrEIA.framework as Ff, FrEIA.modules as Fm
from config import *                           # DEVICE · WAVELET · WM_STRING · WM_STRENGTH

# ───────────────────────────────── INN utils ─────────────────────────────────
def subnet(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Conv2d(c_in, 32, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(32, c_out, 3, padding=1))

def build_inn(C, H, W, blocks):
    nodes = [Ff.InputNode(C, H, W, name='in')]
    for k in range(blocks):
        nodes.append(Ff.Node(nodes[-1], Fm.AllInOneBlock,
                             {"subnet_constructor": subnet},
                             name=f'inn_{k}'))
    nodes.append(Ff.OutputNode(nodes[-1], 'out'))
    return Ff.ReversibleGraphNet(nodes, verbose=False)

def load_net(path):
    ck = torch.load(path, map_location=DEVICE)
    C, H, W = ck['in_shape']
    net = build_inn(C, H, W, ck['num_blocks']).to(DEVICE)
    net.load_state_dict(ck['state_dict']); net.eval()
    return net
# ─────────────────────────────────────────────────────────────────────────────

# 모델 로드
netA = load_net('inn_both.pth')    # LH+HL
netB = load_net('inn_lh.pth')      # LH
netC = load_net('inn_hl.pth')      # HL
netF = load_net('inn_full.pth')    # LL+LH+HL+HH

# 입력 DWT 계수
gray = cv2.cvtColor(cv2.resize(cv2.imread('test_image.jpg'), (256, 256)),
                    cv2.COLOR_BGR2GRAY)
LL, (LH, HL, HH) = pywt.dwt2(gray, WAVELET)
LLn, LHn, HLn, HHn = [x / 255. for x in (LL, LH, HL, HH)]

origA = torch.from_numpy(np.stack([LHn, HLn], 0)).unsqueeze(0).float().to(DEVICE)
inp_LH = torch.from_numpy(np.stack([LHn, LHn], 0)).unsqueeze(0).float().to(DEVICE)
inp_HL = torch.from_numpy(np.stack([HLn, HLn], 0)).unsqueeze(0).float().to(DEVICE)
origF = torch.from_numpy(np.stack([LLn, LHn, HLn, HHn], 0)).unsqueeze(0).float().to(DEVICE)


# 워터마크 두 ½ 생성
bits = ''.join(f'{ord(c):08b}' for c in WM_STRING)
vec = (np.array(list(bits), np.float32) * 2 - 1) * WM_STRENGTH
wm_half1 = torch.tensor(vec.mean() / 2, device=DEVICE)
wm_half2 = torch.tensor(vec.mean() / 2, device=DEVICE)

@torch.no_grad()
def apply_watermark(z, chL, chH, half1, half2):
    """z: (B,C,H,W) → LH·HL 채널에 half1·half2 삽입"""
    z_mod = z.clone()
    z_mod[:, chL] += half1
    z_mod[:, chH] += half2
    return z_mod

# ───────── Stego 생성 ─────────
with torch.no_grad():
    # A
    zA, _ = netA(origA)
    stegoA = netA(apply_watermark(zA, 0, 1, wm_half1, wm_half2), rev=True)

    # B+C
    zB, _ = netB(inp_LH)
    zC, _ = netC(inp_HL)
    ste_LH = netB(apply_watermark(zB, 0, 1, wm_half1, torch.tensor(0., device=DEVICE)), rev=True)
    ste_HL = netC(apply_watermark(zC, 0, 1, torch.tensor(0., device=DEVICE), wm_half2), rev=True)
    if isinstance(ste_LH, tuple): ste_LH = ste_LH[0]
    if isinstance(ste_HL, tuple): ste_HL = ste_HL[0]

    LH_st = ste_LH[0, 0].detach().cpu().numpy()   # (128,128)
    HL_st = ste_HL[0, 0].detach().cpu().numpy()

    
    # stack → (1,2,H,W) → GPU
    stegoBC_coeff = (
        torch.from_numpy(np.stack([LH_st, HL_st], 0))  # (2,128,128)
            .unsqueeze(0)                            # (1,2,128,128)
            .float()                                 # dtype 통일
            .to(DEVICE)
    )

    # Full (채널 idx 1,2 → LH,HL)
    zF, _ = netF(origF)
    stegoF = netF(apply_watermark(zF, 1, 2, wm_half1, wm_half2), rev=True)

# ───────── 계수 → spatial 변환 ─────────
def coeff2img(coeff):
    if isinstance(coeff, tuple):
        coeff = coeff[0]          # (1,C,H,W)

    t = coeff[0]                  # (C,H,W)

    if t.shape[0] == 2:           # 2-채널 케이스 ------------------
        LL_, _  = pywt.dwt2(gray / 255., WAVELET)          # 원본 LL
        LH_, HL_ = (c.detach().cpu().squeeze().numpy() for c in t)
        HH_      = np.zeros_like(LH_)      # ✨ 핵심: 0-array 추가
    else:                                      # 4-채널 케이스 ------
        LL_, LH_, HL_, HH_ = (c.detach().cpu().squeeze().numpy()
                              for c in t)

    return pywt.idwt2((LL_, (LH_, HL_, HH_)), WAVELET)

recA = coeff2img(stegoA)
recBC = coeff2img(stegoBC_coeff)
recF = coeff2img(stegoF)

# ───────── 공통 지표 함수 ─────────
def to_u8(img): return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)

def quality_metrics(name, rec):
    p  = psnr(gray, to_u8(rec), data_range=255)
    s  = ssim(gray, to_u8(rec), data_range=255, win_size=7)
    enc = cv2.imencode('.jpg', to_u8(rec), [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
    jpeg = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    p_jpeg = psnr(gray, jpeg, data_range=255)
    noisy = np.clip(rec + np.random.normal(0, 5/255, rec.shape), 0, 1)
    p_gn = psnr(gray, to_u8(noisy), data_range=255)
    print(f"{name:<8}▶ PSNR {p:6.2f} dB | SSIM {s:.4f} | "
          f"JPEG90 {p_jpeg:5.2f} dB | GN σ5 {p_gn:5.2f} dB")
    return rec

print("\n── Quality Metrics (LH·HL 두 ½ 삽입) ─────────────────────")
rec_dict = {'A-only': quality_metrics('A-only', recA),
            'B+C':    quality_metrics('B+C',   recBC),
            'Full':   quality_metrics('Full',  recF)}

# ───────── Bit-ACC / BER / NC / IF-RMSE ─────────
@torch.no_grad()
def wm_accuracy(model, inp, chL, chH, name, combine=None):
    """
    model  : INN (netA / netB / …)
    inp    : (1,C,H,W) 계수 텐서
    chL,H  : LH, HL 이 위치한 채널 index
    name   : 표시에 쓸 라벨
    combine: ('LH','HL') 두 개 결과를 합쳐 원본 워터마크와 비교할 때 사용
    """
    # 1)  ±1 비트  랜덤 워터마크 → half 삽입
    wm_bits = np.random.randint(0, 2, (2, 128, 128)).astype(np.float32)
    wm_t    = torch.tensor((wm_bits*2-1)*WM_STRENGTH, device=DEVICE)

    # 2) 순전파 / 삽입
    z0, _ = model(inp)                         # z0  (1,C,H,W)
    z_st = z0.clone()
    z_st[:, chL] += wm_t[0]                    # LH 채널
    z_st[:, chH] += wm_t[1]                    # HL 채널
    stego, _   = model(z_st, rev=True)
    z_back, _  = model(stego)

    # 3) 복호
    wm_rec = (z_back - z0).sign().cpu().numpy()  # (1,C,H,W)
    if wm_rec.shape[1] == 4:                     # Full 모델 → LH,HL 슬라이스
        wm_rec = wm_rec[:, 1:3]

    wm_rec = wm_rec.squeeze(0)                   # (2,H,W)
    wm_gt  = wm_t.sign().cpu().numpy()

    # 4) 지표
    BER = (wm_rec != wm_gt).mean()
    ACC = 1 - BER
    NC  = ( (wm_rec>0).astype(np.uint8) * (wm_gt>0).astype(np.uint8) ).sum() / wm_gt.size
    IF  = float(np.sqrt(np.mean((coeff2img(stego) - gray/255.)**2)))

    print(f"{name:<8}▶ ACC {ACC*100:6.2f}% | BER {BER*100:5.2f}% "
          f"| NC {NC:.3f} | IF-RMSE {IF:.5f}")

    # 5) B·C 결과를 합쳐서 한 줄로 비교하고 싶을 때 (combine 인자 사용)
    if combine is not None:
        combine.append((wm_rec, wm_gt))


print("\n── Watermark Extraction Metrics ────────────────────────")
combine_BC = []              # LH·HL 따로 구한 뒤 마지막에 합산할 버퍼

wm_accuracy(netA, origA, 0, 1, 'A-only')
wm_accuracy(netB, inp_LH, 0, 1, 'B-LH', combine_BC)
wm_accuracy(netC, inp_HL, 0, 1, 'C-HL', combine_BC)
wm_accuracy(netF, origF, 1, 2, 'Full')

# ── B + C 결합 워터마크 정확도 한 줄 추가 ─────────────────────
if combine_BC:
    wm_rec_BC = np.stack([combine_BC[0][0], combine_BC[1][0]], 0)
    wm_gt_BC  = np.stack([combine_BC[0][1], combine_BC[1][1]], 0)
    BER = (wm_rec_BC != wm_gt_BC).mean()
    ACC = 1 - BER
    NC  = ( (wm_rec_BC>0).astype(np.uint8) * (wm_gt_BC>0).astype(np.uint8)
          ).sum() / wm_gt_BC.size
    print(f"{'B+C':<8}▶ ACC {ACC*100:6.2f}% | BER {BER*100:5.2f}% "
          f"| NC {NC:.3f} | IF-RMSE –")

# ───────── 시각화 ─────────
titles = ('Cover', 'Stego-A', 'Stego-BC', 'Stego-Full')
cover_coeff = torch.from_numpy(np.stack([LHn, HLn], 0)).unsqueeze(0).float() 
coeffs  = (cover_coeff, stegoA, stegoBC_coeff, stegoF)

rows,cols = 4,4
plt.figure(figsize=(3.6*cols, 8))
for i,(c,ttl) in enumerate(zip(coeffs,titles)):
    if isinstance(c, tuple): c=c[0]
    if c.shape[1]==2:
        LLv = pywt.dwt2(gray/255., WAVELET)[0]; HHv=None
        LHv,HLv = c[0,0].cpu(), c[0,1].cpu()
    else:
        LLv,LHv,HLv,HHv = [ch.cpu() for ch in c[0]]
    recon = pywt.idwt2((LLv,(LHv,HLv,HHv)), WAVELET)
    diff  = np.abs(recon - gray/255.)

    plt.subplot(rows,cols,1+i); plt.imshow(recon,cmap='gray'); plt.axis('off'); plt.title(ttl)
    if ttl=='Cover': continue
    plt.subplot(rows,cols,1+cols+i);   plt.imshow(diff*10,cmap='gray');     plt.axis('off'); plt.title('Abs×10')
    plt.subplot(rows,cols,1+2*cols+i); plt.imshow(diff,cmap='gray');      plt.axis('off'); plt.title('Abs')
    plt.subplot(rows,cols,1+3*cols+i); plt.imshow(diff,cmap='gray');    plt.axis('off'); plt.title('Residual')

plt.suptitle('Cover & Stego Visualisation (LH·HL ½-Watermark)', fontsize=15, y=1.02)
plt.tight_layout(); plt.show()
# ────────────────────────────────────────────────────────────
