"""
필요 패키지:
pip install pillow numpy matplotlib pywavelets torch FrEIA
"""

import numpy as np, pywt, matplotlib.pyplot as plt, cv2, torch
import torch.nn as nn, torch.optim as optim
import FrEIA.framework as Ff, FrEIA.modules as Fm
from pathlib import Path


# ────────────────────────────────────────────────
# 1. 입력 이미지 & 워터마크 준비
# ────────────────────────────────────────────────
IMG_PATH = 'test_image.jpg'                       # 64×64 회색으로 리사이즈할 원본
assert Path(IMG_PATH).exists(), "이미지를 준비하세요!"

# (1) 64×64 회색 이미지
orig_img = cv2.resize(cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE),
                      (64,64)).astype(np.float32) / 255.

# (2) 워터마크 문자열을 8×8=64bit 로 변환
wm_str    = "seo12345"                      # <= 원하는 문자열
bin_str   = ''.join(f'{ord(c):08b}' for c in wm_str)[:64].ljust(64,'0')
wm_bits   = np.array([int(b) for b in bin_str]).reshape(8,8)*255

# ────────────────────────────────────────────────
# 2. DWT ⟹ HH + 나머지 서브밴드
# ────────────────────────────────────────────────
LL,(LH,HL,HH) = pywt.dwt2(orig_img,'haar')   # HH: 32×32
HH_vec  = torch.tensor(HH.flatten(), dtype=torch.float32)      # 1024
wm_vec  = torch.tensor(wm_bits.flatten()/255., dtype=torch.float32)  # 64
x_vec   = torch.cat([HH_vec, wm_vec])                           # 1088


# ────────────────────────────────────────────────
# 3. INN 정의 (1088→1088, 아주 얕음)
# ────────────────────────────────────────────────
def subnet(d_in,d_out):
    return nn.Sequential(nn.Linear(d_in,128), nn.ReLU(), nn.Linear(128,d_out))

nodes=[Ff.InputNode(1088,name='in')]
nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                     {'subnet_constructor':subnet,'clamp':1.0}, name='cb'))
nodes.append(Ff.OutputNode(nodes[-1], name='out'))
inn = Ff.ReversibleGraphNet(nodes)


# ────────────────────────────────────────────────
# 4. self-reconstruction 학습
# ────────────────────────────────────────────────
opt, loss_fn = optim.Adam(inn.parameters(),1e-3), nn.MSELoss()
x_batch = x_vec.unsqueeze(0).repeat(64,1)      # (batch=64, 1088)

for ep in range(400):
    out   = inn(x_batch)[0]
    recon = inn(out, rev=True)[0]
    loss  = loss_fn(recon, x_batch)
    opt.zero_grad(); loss.backward(); opt.step()
    if ep%100==0:
        print(f'epoch {ep:3d}  loss={loss.item():.3e}')

# ────────────────────────────────────────────────
# 5. 인코딩(삽입) : HH → y 로 교체 → 스테고 이미지
# ────────────────────────────────────────────────
out_single = inn(x_vec.unsqueeze(0))[0]        # (1,1088)
y_HH, z_code = out_single[0,:1024], out_single[0,1024:]

HH_stego = y_HH.reshape(32,32).detach().numpy()
stego_img = pywt.idwt2((LL,(LH,HL,HH_stego)),'haar')
stego_img = np.clip(stego_img,0,1)

# ────────────────────────────────────────────────
# 6. 디코딩(추출) : 스테고 HH + z_code → 역변환
# ────────────────────────────────────────────────
#   ① 스테고에서 HH' 추출
_,(_,_,HH_s) = pywt.dwt2(stego_img,'haar')
HH_s_vec = torch.tensor(HH_s.flatten(), dtype=torch.float32)

#   ② HH'‖z_code 를 역방향으로
x_cat   = torch.cat([HH_s_vec, z_code])
x_back  = inn(x_cat.unsqueeze(0), rev=True)[0]

#   ③ 복원 워터마크
wm_rec = (x_back[0,1024:] > 0.5).view(8,8).numpy()
print("워터마크 일치? ", np.array_equal(wm_bits, wm_rec*255))

# ────────────────────────────────────────────────
# 7. 시각 결과
# ────────────────────────────────────────────────
plt.figure(figsize=(12,4))
titles_imgs = [("Original", orig_img),
               ("Stego",    stego_img),
               ("|diff|×5", np.abs(orig_img-stego_img)*5),
               ("WM in",    wm_bits),
               ("WM out",   wm_rec*255)]
for i,(t,img) in enumerate(titles_imgs,1):
    plt.subplot(1,5,i); plt.imshow(img, cmap='gray'); plt.title(t); plt.axis('off')
plt.tight_layout(); plt.show()
