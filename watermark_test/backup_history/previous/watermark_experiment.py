import os, cv2, pywt, torch, numpy as np, matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import FrEIA.framework as Ff, FrEIA.modules as Fm
from config import *

# model util
def subnet(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Conv2d(c_in, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, c_out, 3, padding=1)
    )

def build_inn(C, H, W, blocks):
    n = [Ff.InputNode(C, H, W, name = 'in')]
    for k in range(blocks):
        n.append(Ff.Node(n[-1], Fm.AllInOneBlock,
                         {"subnet_constructor": subnet},
                         name=f'inn_{k}'))
    n.append(Ff.OutputNode(n[-1], name='out'))
    return Ff.ReversibleGraphNet(n, verbose=False)

def load_net(pth):
    ck = torch.load(pth, map_location=DEVICE)
    C, H, W = ck['in_shape']
    blocks = ck['num_blocks']
    net = build_inn(C, H, W, blocks).to(DEVICE)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return net

# prepare
netA = load_net('inn_both.pth')
netB = load_net('inn_lh.pth')
netC = load_net('inn_hl.pth')

img_path = 'test_image.jpg'
gray = cv2.cvtColor(cv2.resize(cv2.imread(img_path), (256, 256)), cv2.COLOR_BGR2GRAY)
LH, HL = pywt.dwt2(gray, WAVELET)[1][:2]
orig = torch.tensor(np.stack([LH/255, HL/255], 0)).unsqueeze(0).float().to(DEVICE)

bits = ''.join(f'{ord(c):08b}' for c in WM_STRING)
vec = (np.array(list(bits), np.float32)*2-1) * WM_STRENGTH
wm_val = torch.tensor(vec.mean()).to(DEVICE)


# A senario 
zA, _ = netA(orig)
stegoA = netA(zA + wm_val, rev = True)


# B+C senario
half = wm_val / 2
z_mix, _ = netA(orig)                  # z는 (B, 2, H, W)
z_mix[:, 0] += half                    # LH 쪽
z_mix[:, 1] += half                    # HL 쪽
stegoBC = netA(z_mix, rev=True)

# ─────────── 품질 지표 (PSNR·SSIM + 5개 추가) ───────────
img_o  = orig[0,0].cpu().numpy()
img_A  = stegoA[0][0].detach().cpu().numpy()
img_BC = stegoBC[0][0].detach().cpu().numpy()

img_o,img_A,img_BC=[np.squeeze(x) for x in (img_o,img_A,img_BC)]
o255,A255,BC255=[(x*255).round().astype(np.uint8) for x in (img_o,img_A,img_BC)]

A255  = A255[0]       
BC255 = BC255[0]

psnrA, psnrBC = [psnr(o255, x, data_range=255) for x in (A255, BC255)]
ssimA,ssimBC=[ssim(o255,x,data_range=255,win_size=7) for x in (A255,BC255)]

# ---- 추가 지표 ----------------------------------------
wm_bits = np.random.randint(0,2,(2,128,128)).astype(np.float32)
wm_t    = torch.tensor((wm_bits*2-1)*WM_STRENGTH,device=DEVICE)  # (2,128,128)

z_orig,_   = netA([orig])
z_stego    = z_orig + wm_t.unsqueeze(0)
stego,_    = netA([z_stego],rev=True)
z_rec,_    = netA([stego])
wm_rec = (z_rec - z_orig).sign().detach().cpu().numpy().astype(np.int8)[0]
wm_pred    = (wm_rec>0).astype(np.uint8)

BER      = (wm_pred!=wm_bits).mean()          # Bit-Error-Rate (0=완벽)
ACC      = 1-BER                              # Bit Accuracy   (1=완벽)
NC       = (wm_pred*wm_bits).sum()/wm_bits.size
IF_RMSE  = float(np.sqrt(np.mean((img_o-img_A)**2)))

# JPG-90% 강인성
enc=cv2.imencode('.jpg',A255,[cv2.IMWRITE_JPEG_QUALITY,90])[1]
jpeg=cv2.imdecode(enc,cv2.IMREAD_GRAYSCALE)
psnr_jpeg = psnr(o255, jpeg, data_range=255)

def to_2d(arr):
    arr = np.asarray(arr)
    while arr.ndim > 2:
        arr = arr[0]          
    return np.squeeze(arr)    

# Gaussain-Noise(5) 강인성
sigma=5/255
noise     = np.random.normal(0, sigma, img_A.shape)  
noisy_img = np.clip(img_A + noise, 0, 1)
o255_u8   = to_2d(o255)                         
noisy_u8  = to_2d((noisy_img*255).round().astype(np.uint8))

psnr_gn = psnr(o255_u8, noisy_u8, data_range=255)

print(f"A-only ▶ PSNR {psnrA:6.2f} dB | SSIM {ssimA:.4f}")
print(f"B+C    ▶ PSNR {psnrBC:6.2f} dB | SSIM {ssimBC:.4f}")
print(f"Bit-ACC {ACC*100:5.2f}%  | BER {BER*100:5.2f}%  | NC {NC:.3f}")
print(f"IF(RMSE) {IF_RMSE:.5f}")
print(f"JPEG90-PSNR {psnr_jpeg:.2f} dB | GN(σ=5)-PSNR {psnr_gn:.2f} dB")

# ───────────── 시각화 (역 DWT 복원) ─────────────
titles   = ('Cover', 'Stego-A', 'Stego-BC', 'Stego-Full')
tensors  = (orig,    stegoA,    stegoBC,    stegoC)

plt.figure(figsize=(12, 8))

for i, (t, ttl) in enumerate(zip(tensors, titles), 1):
    if isinstance(t, tuple): t = t[0]

    # 채널 개수에 따라 처리
    if t.shape[1] == 4:
        LL, LH, HL, HH = [c.detach().cpu().numpy() for c in t[0]]
    else:
        LH = t[0, 0].detach().cpu().numpy()
        HL = t[0, 1].detach().cpu().numpy()
        LL, _ = pywt.dwt2(gray / 255.0, WAVELET)

    # 역변환 및 차이 계산
    recon = pywt.idwt2((LL, (LH, HL, None)), WAVELET)
    diff = np.abs(recon - gray / 255.0)
    residual = diff.copy()

    # 1. 복원된 이미지
    plt.subplot(4, 4, i)
    plt.imshow(recon, cmap='gray', vmin=0, vmax=1)
    plt.title(ttl); plt.axis('off')

    # 나머지는 Cover 제외하고 시각화
    if ttl == 'Cover':
        continue

    # 2. 절대 오차 ×10
    plt.subplot(4, 4, i + 4)
    plt.imshow(diff * 10, cmap='hot', vmin=0, vmax=1)
    plt.title('Abs×10'); plt.axis('off')

    # 3. 절대 오차 (히트맵)
    plt.subplot(4, 4, i + 8)
    plt.imshow(diff, cmap='magma', vmin=0, vmax=0.1)
    plt.title('Abs'); plt.axis('off')

    # 4. Residual Map
    plt.subplot(4, 4, i + 12)
    plt.imshow(residual, cmap='inferno', vmin=0, vmax=0.1)
    plt.title('Residual'); plt.axis('off')

plt.tight_layout()
plt.suptitle("Stego Visualization + Embedding Residual Map", fontsize=14, y=1.03)
plt.show()
