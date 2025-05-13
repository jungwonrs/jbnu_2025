import numpy as np, cv2, pywt, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim, FrEIA.framework as Ff, FrEIA.modules as Fm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from pathlib import Path

# ---------- 1. 이미지 & 워터마크 ------------------------------------
IMG_PATH = 'test_image.jpg'                              # 64x64 로 줄여 쓸 파일
assert Path(IMG_PATH).exists(), "이미지 준비!"

orig_img = cv2.resize(cv2.imread(IMG_PATH,0),(64,64)).astype(np.float32)/255.
wm_str   = "seo12345"
bin_str  = ''.join(f'{ord(c):08b}' for c in wm_str).ljust(64,'0')

'''
이미지 형태의 워터마크 데이터로 처리하기 위해서 배열화 진행
'''
wm_bits  = np.fromiter(bin_str, int).reshape(8,8)*255

# ---------- 2. DWT  --------------------------------------------------
LL,(LH,HL,HH) = pywt.dwt2(orig_img,'haar')
x_vec = torch.tensor(np.hstack([HH.flatten(), wm_bits.flatten()/255.]),
                     dtype=torch.float32)          # 1024+64=1088

# -------- 3. 초소형 INN (수정 후) -------------------------------
def subnet(din, dout):
    return nn.Sequential(
        nn.Linear(din, 128),
        nn.ReLU(),
        nn.Linear(128, dout)
    )

nodes = [Ff.InputNode(1088, name='in')]
nodes.append(
    Ff.Node(
        nodes[-1],                          # ← None 대신 마지막 노드
        Fm.GLOWCouplingBlock,
        {'subnet_constructor': subnet, 'clamp': 1.0},
        name='cb'
    )
)
nodes.append(Ff.OutputNode(nodes[-1], name='out'))

inn = Ff.ReversibleGraphNet(nodes)

# ---------- 4. self-reconstruction 학습 ------------------------------
batch = x_vec.unsqueeze(0).repeat(64,1)
opt,loss_fn = optim.Adam(inn.parameters(),1e-3), nn.MSELoss()
for ep in range(300):
    out   = inn(batch)[0]
    recon = inn(out, rev=True)[0]
    loss  = loss_fn(recon, batch)
    opt.zero_grad(); loss.backward(); opt.step()
    if ep%100==0: print(f'epoch {ep:3d}  loss={loss.item():.3e}')

# --- 5. 인코딩(삽입) ---
out = inn(x_vec.unsqueeze(0))[0]              # (1,1088) Autograd-attached
HH_stego = out[0, :1024].detach().view(32, 32).numpy()  # ← detach!
stego = pywt.idwt2((LL, (LH, HL, HH_stego)), 'haar')
stego = np.clip(stego, 0, 1)

# ---------- 6. 추출 --------------------------------------------------
def decode_watermark(bits):                        # 64bit → 문자열
    chars=[chr(int(''.join(str(int(b)) for b in bits[i:i+8]),2)) for i in range(0,64,8)]
    return ''.join(chars).rstrip('\x00')

x_back = inn(out.detach(), rev=True)[0]       # rev 방향엔 그래프 끊은 텐서 전달
wm_rec_bits = (x_back[0, 1024:].detach() > .5).cpu().numpy().astype(int)
wm_rec = wm_rec_bits.reshape(8,8)*255
print("Recovered watermark:", decode_watermark(wm_rec_bits))

# ---------- 7. 품질 수치 ---------------------------------------------
print("PSNR:", psnr(orig_img, stego))
print("SSIM:", ssim(orig_img, stego, data_range=1.0))      # ★ 수정

# ---------- 8. 시각화 ------------------------------------------------
plt.figure(figsize=(12,4))
def show(idx,title,img,vmin=0,vmax=1):
    plt.subplot(1,5,idx); plt.imshow(img,cmap='gray',vmin=vmin,vmax=vmax); plt.title(title); plt.axis('off')
show(1,'Original', orig_img)
show(2,'Stego',    stego)
show(3,'|diff|×5', np.abs(orig_img-stego)*5)
show(4,'WM in',    wm_bits, vmin=0, vmax=255)
show(5,'WM out',   wm_rec , vmin=0, vmax=255)
plt.tight_layout(); plt.show()
