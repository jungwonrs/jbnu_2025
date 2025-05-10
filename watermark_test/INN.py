"""
❑ 목적: 64×64 회색 이미지에 8×8 비트 워터마크 삽입→추출
   └ DWT(Haar) HH밴드만 변형
   └ 초소형 INN(가역망) 으로 재구성 보장
❑ 의존: numpy, torch, pywavelets, matplotlib, pillow, FrEIA
"""

import numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt, pywt
from PIL import Image
import FrEIA.framework as Ff, FrEIA.modules as Fm

# ────────────────────────────────────────────────
# 0. 도우미 함수
# ────────────────────────────────────────────────
def to_img(vec):  # 1-D 텐서 → 32×32
    return vec.detach().cpu().numpy().reshape(32, 32)

def show(title,img,idx):
    plt.subplot(2,3,idx); plt.imshow(img,cmap='gray'); plt.title(title); plt.axis('off')

# ────────────────────────────────────────────────
# 1. 데이터 준비 (64×64 회색 이미지 + 8×8 워터마크)
# ────────────────────────────────────────────────
orig_img = np.array(Image.open('test_image.jpg').resize((64,64)).convert('L'))/255.
wm_bits  = np.random.randint(0,2,(8,8))*255       # 0/255 비트

# 1-level Haar DWT
LL,(LH,HL,HH) = pywt.dwt2(orig_img,'haar')        # HH: 32×32

HH_vec = torch.tensor(HH.flatten(),dtype=torch.float32)       # (1024,)
wm_vec = torch.tensor(wm_bits.flatten()/255.,dtype=torch.float32) # (64,)
x_vec  = torch.cat([HH_vec, wm_vec])                          # (1088,)

# ────────────────────────────────────────────────
# 2. 초소형 INN 정의 (Input=1088, Output=1088)
# ────────────────────────────────────────────────
def subnet(d_in,d_out):
    return nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, d_out))

nodes=[Ff.InputNode(1088,name='in')]
nodes.append(Ff.Node(nodes[-1],Fm.GLOWCouplingBlock,
                     {'subnet_constructor':subnet,'clamp':1.0},name='cb'))
nodes.append(Ff.OutputNode(nodes[-1],name='out'))
inn = Ff.ReversibleGraphNet(nodes)

# ────────────────────────────────────────────────
# 3. self-reconstruction 학습 (x → INN → INN⁻¹ → x)
# ────────────────────────────────────────────────
opt, loss_fn = optim.Adam(inn.parameters(),1e-3), nn.MSELoss()
x_batch = x_vec.unsqueeze(0).repeat(64,1)   # batch=64

for ep in range(300):
    out = inn(x_batch)[0]                   # forward
    recon = inn(out, rev=True)[0]           # inverse
    loss = loss_fn(recon, x_batch)
    opt.zero_grad(); loss.backward(); opt.step()
    if ep%100==0: print(f'epoch {ep:3d}  loss={loss.item():.3e}')

# ────────────────────────────────────────────────
# 4. 인코딩(삽입)
# ────────────────────────────────────────────────
out_single = inn(x_vec.unsqueeze(0))[0]     # (1,1088)
y_HH       = out_single[0,:1024]            # 스테고용 HH
z_stored   = out_single[0,1024:]            # 저장-예정 워터마크 code

HH_stego   = to_img(y_HH)                   # 32×32
stego_img  = pywt.idwt2((LL,(LH,HL,HH_stego)),'haar')

# ────────────────────────────────────────────────
# 5. 디코딩(추출)
# ────────────────────────────────────────────────
x_back = inn(out_single, rev=True)[0]
wm_rec = (x_back[0,1024:] > 0.5).view(8,8).numpy()

# ────────────────────────────────────────────────
# 6. 결과 시각화
# ────────────────────────────────────────────────
plt.figure(figsize=(9,6))
show('Original', orig_img,1)
show('Stego',    stego_img,2)
show('|diff|',   np.abs(orig_img-stego_img)*5,3)   # *5 ⇢ 차이 강조
show('WM in',  wm_bits,4)
show('WM out', wm_rec*255,5)
plt.tight_layout(); plt.show()

print('워터마크 일치 여부:', np.array_equal(wm_bits, wm_rec*255))
