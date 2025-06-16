# ─────────── model_experiment4.py  (A·B·C 모델) ───────────
import os, random, time, cv2, pywt, torch, numpy as np
import torch.nn as nn, torch.optim as optim
import FrEIA.framework as Ff, FrEIA.modules as Fm
from pycocotools.coco import COCO
from tqdm import tqdm
from config import *                       # DEVICE, LR, BLOCKS, … 그대로 사용
import torch.nn.functional as F         
IMP_W = None          # train() 안에서 lazy-init


# ────────── util ──────────────────────────────────────────
def dwt_coeffs(gray):
    """LL,LH,HL,HH (float32) 반환"""
    LL, (LH, HL, HH) = pywt.dwt2(gray, WAVELET)
    return [c.astype(np.float32) for c in (LL, LH, HL, HH)]

def load_tensors(split:str):
    """
    split:
      'both' → [LH,HL] 2채널  (A)
      'lh'   → [LH,LH] 2채널  (B-LH)
      'hl'   → [HL,HL] 2채널  (B-HL)
      'full' → [LL,LH,HL,HH] 4채널  (C)
    """
    coco  = COCO(TRAIN_JSON)
    ids   = random.sample(coco.getImgIds(), N_IMG)
    data  = []

    for i in tqdm(ids, desc=f'COCO-{split}'):
        p = os.path.join(TRAIN_DIR, coco.loadImgs(i)[0]['file_name'])
        g = cv2.cvtColor(cv2.resize(cv2.imread(p), (256, 256)), cv2.COLOR_BGR2GRAY)
        LL, LH, HL, HH = dwt_coeffs(g)

        if   split == 'both':
            x = np.stack([LH/255, HL/255], 0)
        elif split == 'lh':
            x = np.stack([LH/255, LH/255], 0)
        elif split == 'hl':
            x = np.stack([HL/255, HL/255], 0)
        elif split == 'full':
            x = np.stack([LL/255, LH/255, HL/255, HH/255], 0)
        else:
            raise ValueError(f'unknown split {split}')
        data.append(torch.from_numpy(x).float())

    return torch.stack(data).to(DEVICE)

# ────────── INN 정의 ─────────────────────────────────────
def subnet(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, c_out, 3, padding=1)
    )

def build_inn(C, H, W):
    nodes = [Ff.InputNode(C, H, W, name='in')]

    # 1채널인 경우만 HaarDownsampling 유지(기존 로직)
    if C == 1:
        nodes.append(Ff.Node(nodes[-1],
                             Fm.HaarDownsampling,
                             {"rebalance": False, "order_by_wavelet": True},
                             name='haar'))

    for k in range(BLOCKS):
        nodes.append(Ff.Node(nodes[-1],
                             Fm.AllInOneBlock,
                             {"subnet_constructor": subnet},
                             name=f'inn_{k}'))
    nodes.append(Ff.OutputNode(nodes[-1], name='out'))
    return Ff.ReversibleGraphNet(nodes, verbose=False)

# ────────── 학습 루프 ─────────────────────────────────────
# ---------------- importance utils ----------------
def sobel_kernel(ksize=3):
    if ksize == 3:
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32)
    else:               # 5×5 Scharr-like
        kx = torch.tensor([[-2,-1,0,1,2],
                           [-3,-2,0,2,3],
                           [-4,-3,0,3,4],
                           [-3,-2,0,2,3],
                           [-2,-1,0,1,2]], dtype=torch.float32)
    ky = kx.t()
    return kx.view(1,1,*kx.shape), ky.view(1,1,*ky.shape)

@torch.no_grad()
def build_importance_map(batch: torch.Tensor, k=IMP_SOBEL_K, eps=1e-6):
    """
    batch: (B, C, H, W)  –– **실제 입력 텐서로부터** grad-mag
    반환: (1,1,H,W)  (모든 채널·샘플 평균)
    """
    # luminance proxy  ▶  (B,1,H,W)
    if batch.size(1) >= 3:
        lum = batch[:,1:3].mean(1, keepdim=True)  # LH+HL 평균
    else:
        lum = batch.mean(1, keepdim=True)         # (B,1,…)

    kx,ky = sobel_kernel(k); kx,ky = kx.to(batch), ky.to(batch)
    gx = F.conv2d(lum, kx, padding=k//2)
    gy = F.conv2d(lum, ky, padding=k//2)
    grad = torch.sqrt(gx.pow(2)+gy.pow(2)+eps)    # (B,1,H,W)

    imp = 1. - grad / (grad.amax(dim=(-3,-2,-1), keepdim=True) + eps)
    return imp.mean(0, keepdim=True)  # (1,1,H,W)

def train(tag: str, tensors: torch.Tensor):
    loader = torch.utils.data.DataLoader(tensors, batch_size=BATCH, shuffle=True)
    C, H, W = tensors.shape[1:]
    net = build_inn(C, H, W).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=LR)

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tot = 0

        for x in loader:
            if isinstance(x, (list, tuple)):
                x = x[0]

            # Tensor인지 확인 (선택적 안전장치)
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Expected tensor, got {type(x)}")

            z, logJ = net(x)

            # --- importance map (gradient-mag) ---
            IMP_W = build_importance_map(x, k=IMP_SOBEL_K).to(DEVICE) 

            loss_img = (IMP_W * z).pow(2).mean()
            loss = LAM_Z * IMP_GAIN * loss_img - LAM_J * logJ.mean()

            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)

        print(f'[{tag}] Ep{ep}/{EPOCHS} NLL={tot/len(tensors):.4f}')
    elapsed = time.time() - t0
    print(f'[{tag}] {elapsed:.1f}s')
    
    
    folder = os.path.join(MODEL_DIR, tag)
    os.makedirs(folder, exist_ok=True)

    torch.save({
        'state_dict': net.state_dict(),
        'in_shape': (C, H, W),
        'num_blocks': BLOCKS
    }, os.path.join(folder, f'inn_{tag}.pth'))
    

    # ⏱️ 시간도 별도 기록
    with open(os.path.join(folder, 'train_time.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{elapsed:.3f} sec\n")

    return net

# ────────── main ─────────────────────────────────────────
if __name__ == '__main__':
    DS_BOTH = load_tensors('both')   # A-모델
    DS_LH   = load_tensors('lh')     # B-LH
    DS_HL   = load_tensors('hl')     # B-HL
    DS_FULL = load_tensors('full')   # C-모델 (LL+LH+HL+HH)

    train('both',  DS_BOTH)   # A
    train('lh',    DS_LH)     # B1
    train('hl',    DS_HL)     # B2
    train('full',  DS_FULL)   # C  ← 새로 추가
