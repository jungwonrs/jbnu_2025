# ─────────── model_experiment4.py  (A·B·C 모델) ───────────
import os, random, time, cv2, pywt, torch, numpy as np
import torch.nn as nn, torch.optim as optim
import FrEIA.framework as Ff, FrEIA.modules as Fm
from pycocotools.coco import COCO
from tqdm import tqdm
from config import *                       # DEVICE, LR, BLOCKS, … 그대로 사용

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
def train(tag:str, tensors:torch.Tensor):
    loader = torch.utils.data.DataLoader(tensors, batch_size=BATCH, shuffle=True)
    C, H, W = tensors.shape[1:]
    net = build_inn(C, H, W).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=LR)

    t0=time.time()
    for ep in range(1, EPOCHS+1):
        tot=0
        for x in loader:
            z, logJ = net(x)
            loss = LAM_Z * z.pow(2).mean() - LAM_J * logJ.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        print(f'[{tag}] Ep{ep}/{EPOCHS} NLL={tot/len(tensors):.4f}')
    print(f'[{tag}] {time.time()-t0:.1f}s')

    torch.save({'state_dict': net.state_dict(),
                'in_shape': (C, H, W),
                'num_blocks': BLOCKS},
               os.path.join(MODEL_DIR, f'inn_{tag}.pth'))
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
