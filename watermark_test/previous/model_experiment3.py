import os, random, time, cv2, pywt, torch, numpy as np
import torch.nn as nn, torch.optim as optim
import FrEIA.framework as Ff, FrEIA.modules as Fm
from pycocotools.coco import COCO
from tqdm import tqdm
from config import *

# util
def dwt(gray):
    _, (LH, HL, _) = pywt.dwt2(gray, WAVELET)
    return LH.astype(np.float32), HL.astype(np.float32)

def load_tensors(split):
    coco = COCO(TRAIN_JSON)
    ids = random.sample(coco.getImgIds(), N_IMG)
    data = []
    for i in tqdm(ids, desc=f'COCO-{split}'):
        p = os.path.join(TRAIN_DIR, coco.loadImgs(i)[0]['file_name'])
        g = cv2.cvtColor(cv2.resize(cv2.imread(p),(256, 256)), cv2.COLOR_BGR2GRAY)
        LH, HL = dwt(g)

        if split == 'both' : 
            x = np.stack([LH/255, HL/255], 0)
        elif split == 'lh' : 
            x = np.stack([LH/255, LH/255], 0)
        else : 
            x = np.stack([HL/255, HL/255], 0)
        data.append(torch.from_numpy(x).float())
    return torch.stack(data).to(DEVICE)


def subnet(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, c_out, 3, padding=1)
    )

def build_inn(C, H, W):
    n = [Ff.InputNode(C, H, W, name = 'in')]

    if C == 1:
        n.append(Ff.Node(n[-1],
                             Fm.HaarDownsampling,
                             {"rebalance": False,
                              "order_by_wavelet": True},
                              name = 'haar'))

    for k in range(BLOCKS):
        n.append(Ff.Node(n[-1], 
                         Fm.AllInOneBlock,
                         {"subnet_constructor":subnet}, 
                         name=f'inn_{k}'))
        
    n.append(Ff.OutputNode(n[-1], name='out'))
    return Ff.ReversibleGraphNet(n, verbose=False)

def train(tag, tensors):
    loader = torch.utils.data.DataLoader(tensors, batch_size=BATCH, shuffle=True)
    C, H, W = tensors.shape[1:]
    net = build_inn(C, H, W).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=LR)
    t0 = time.time()
    for ep in range (1, EPOCHS+1):
        loss_tot = 0
        for x in loader:
            z, logJ = net(x)
            loss = LAM_Z*z.pow(2).mean() - LAM_J*logJ.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_tot += loss.item() * x.size(0)
        print(f'[{tag}] Ep{ep}/{EPOCHS} NLL={loss_tot/len(tensors):.4f}')
    print(f'[{tag}] {time.time()-t0:.1f}s')
    torch.save({'state_dict': net.state_dict(),
                'in_shape': (C, H, W),
                'num_blocks': BLOCKS},
                os.path.join(MODEL_DIR, f'inn_{tag}.pth'))
    return net

if __name__ == "__main__":
    DS_BOTH = load_tensors('both')
    DS_LH = load_tensors('lh')
    DS_HL = load_tensors('hl')
    train('both', DS_BOTH)
    train('lh', DS_LH)
    train('hl', DS_HL)
