# ─────────── model_experiment4.py  (A·B·C 모델) ───────────
import os, random, pywt, torch, numpy as np
import torch.nn as nn, torch.optim as optim
import FrEIA.framework as Ff, FrEIA.modules as Fm
from pycocotools.coco import COCO
from tqdm import tqdm
from config import *                       # DEVICE, LR, BLOCKS, … 그대로 사용
import torch.nn.functional as F         
IMP_W = None          # train() 안에서 lazy-init
LAMBDA_CLEAN = 0.05  
import torchvision.transforms.functional as TF
import random, cv2

IMP_W = None
LAMBDA_CLEAN = 0.05

# ────────── util ──────────────────────────────────────────
def coeff2img_batch(coeff, wavelet):
    coeff = coeff.detach().cpu().numpy()
    imgs  = []
    for arr in coeff:
        if arr.shape[0] == 2:
            LH, HL = arr
            LL = np.zeros_like(LH); HH = np.zeros_like(LH)
        else:
            LL,LH,HL,HH = arr
        img = pywt.idwt2((LL,(LH,HL,HH)), wavelet)
        imgs.append(img.astype(np.float32))
    return np.stack(imgs,0)

def img2coeff_batch(imgs, C, wavelet):
    coeffs = []
    for img in imgs:
        LL,(LH,HL,HH) = pywt.dwt2(img, wavelet)
        if C == 2:
            arr = np.stack([LH/255., HL/255.],0)
        else:
            arr = np.stack([LL/255., LH/255., HL/255., HH/255.],0)
        coeffs.append(arr.astype(np.float32))
    return torch.from_numpy(np.stack(coeffs)).to(DEVICE)

def make_wm_bits(len_bits:int, seed:int|None, shape:tuple[str]):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=len_bits, dtype=np.uint8)
    h, w = shape
    flat = np.repeat(bits, (h*w + len_bits - 1)//len_bits)[:h*w]
    flat = flat.reshape(h, w)
    return bits, torch.from_numpy(flat).float()  # (H,W)

def dwt_coeffs(gray):
    """LL,LH,HL,HH (float32) 반환"""
    LL, (LH, HL, HH) = pywt.dwt2(gray, WAVELET)
    return [c.astype(np.float32) for c in (LL, LH, HL, HH)]

def load_tensors(split:str):
    coco  = COCO(TRAIN_JSON)
    ids   = random.sample(coco.getImgIds(), N_IMG)
    data  = []
    for i in tqdm(ids, desc=f'COCO-{split}'):
        p = os.path.join(TRAIN_DIR, coco.loadImgs(i)[0]['file_name'])
        g = cv2.cvtColor(cv2.resize(cv2.imread(p), (256, 256)), cv2.COLOR_BGR2GRAY)
        LL, LH, HL, HH = dwt_coeffs(g)
        if   split == 'both': x = np.stack([LH/255, HL/255], 0)
        elif split == 'lh':  x = np.stack([LH/255, LH/255], 0)
        elif split == 'hl':  x = np.stack([HL/255, HL/255], 0)
        elif split == 'full':x = np.stack([LL/255, LH/255, HL/255, HH/255], 0)
        else: raise ValueError(f'unknown split {split}')
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
    if C == 1:
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling,
                             {"rebalance": False, "order_by_wavelet": True},
                             name='haar'))
    for k in range(BLOCKS):
        nodes.append(Ff.Node(nodes[-1], Fm.AllInOneBlock,
                             {"subnet_constructor": subnet},
                             name=f'inn_{k}'))
    nodes.append(Ff.OutputNode(nodes[-1], name='out'))
    return Ff.ReversibleGraphNet(nodes, verbose=False)

class SignSTE(torch.autograd.Function):
    """y = sign(x) ;  dy/dx = 1 (STE)"""
    @staticmethod
    def forward(ctx, x):
        return x.sign()
    @staticmethod
    def backward(ctx, g):
        return g          # .clone() 필요 없음
sign_ste = SignSTE.apply

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
    """(B,C,H,W) → (1,1,H,W) luminance gradient‑mag based importance"""
    lum = batch[:,1:3].mean(1, keepdim=True) if batch.size(1)>=3 else batch.mean(1,keepdim=True)
    kx,ky = sobel_kernel(k); kx,ky = kx.to(batch), ky.to(batch)
    gx = F.conv2d(lum, kx, padding=k//2)
    gy = F.conv2d(lum, ky, padding=k//2)
    grad = torch.sqrt(gx.pow(2)+gy.pow(2)+eps)
    imp  = 1. - grad / (grad.amax(dim=(-3,-2,-1), keepdim=True)+eps)
    return imp.mean(0, keepdim=True)

def jpeg_tensor(x_bchw: torch.Tensor, q: int) -> torch.Tensor:
    """
    x_bchw : (B,1,H,W) 0‥1 float
    q      : JPEG quality
    return : (B,1,H,W) 0‥1 float
    """
    bs = x_bchw.size(0)            # h,w 변수 제거 (unused)
    buf = []
    for k in range(bs):
        u8  = (x_bchw[k,0].clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
        enc = cv2.imencode('.jpg', u8, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        dec = cv2.imdecode(enc, 0).astype(np.float32) / 255.
        buf.append(torch.from_numpy(dec))
    return torch.stack(buf, 0).unsqueeze(1).to(x_bchw.device)

def spatial2coeff(img_f: np.ndarray, mode: str) -> torch.Tensor:
    """1-채널 이미지 (0~1) → LH/HL 또는 4-subband tensor (1,C,H,W)"""
    LL,(LH,HL,HH) = pywt.dwt2(img_f * 255., WAVELET)   # ← 콤마 추가
    LL,LH,HL,HH  = [x/255. for x in (LL,LH,HL,HH)]
    if mode == "2":
        arr = np.stack([LH, HL], 0)
    else:
        arr = np.stack([LL, LH, HL, HH], 0)
    return torch.from_numpy(arr)[None].float().to(DEVICE)

def train(tag: str, tensors: torch.Tensor, build_inn):
    # Reset importance map per model
    global IMP_W
    IMP_W = None

    loader = torch.utils.data.DataLoader(tensors, batch_size=BATCH, shuffle=True, drop_last=True)
    C, H, W = tensors.shape[1:]
    net = build_inn(C, H, W).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=LR, betas=(0.5, 0.999))

    _, wm_map = make_wm_bits(WM_LEN, WM_SEED, (H, W))
    wm_sign = (wm_map * 2 - 1).to(DEVICE) * WM_STRENGTH
    wm_ch = list(range(C)) if C == 2 else [1, 2]

    for ep in range(1, EPOCHS + 1):
        phase = 'A' if ep <= EPOCHS_A else 'B'
        epoch_loss = 0.0
        for step, x in enumerate(loader, 1):
            x = x.to(DEVICE)
            z, logJ = net(x)
            loss = LAM_Z * IMP_GAIN * z.pow(2).mean() - LAM_J * logJ.mean()

            if phase == 'A':
                loss += 0.05 * z[:, wm_ch].pow(2).mean()
                if IMP_W is None and step == 1:
                    IMP_W = build_importance_map(x)
            else:
                z_emb = z.clone()
                for ch in wm_ch:
                    imp_scaled = (0.5 + 0.5 * IMP_W.to(DEVICE))
                    wm = wm_sign.unsqueeze(0) * imp_scaled
                    z_emb[:, ch] += wm.squeeze(1)
                x_stego, _ = net(z_emb, rev=True, jac=False)
                clean_loss = LAMBDA_CLEAN * (x_stego - x).abs().mean()
                loss += clean_loss

                r = random.random()
                if r < .33:
                    σ = random.uniform(1.2, 4.0)
                    x_atk = TF.gaussian_blur(x_stego, (int(σ*4+1)|1), σ)
                elif r < .66:
                    q = random.randint(30, 85)
                    x_atk = jpeg_tensor(x_stego[:, :1], q).repeat(1, C, 1, 1)
                else:
                    x_atk = (x_stego + torch.randn_like(x_stego) * 0.01).clamp(0,1)

                z_back, _ = net(x_atk, jac=False)
                scale = SCALE_LOGIT / WM_STRENGTH
                logits = torch.stack([z_back[:, ch]*scale for ch in wm_ch], 1)
                target = (((wm_sign.sign().unsqueeze(0).unsqueeze(1)) + 1)*0.5).expand_as(logits)
                wm_loss = F.binary_cross_entropy_with_logits(logits, target)
                loss += BETA * wm_loss

            opt.zero_grad()
            loss.backward()
            # gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item() * x.size(0)

        print(f"[{tag}] Ep{ep:03d}/{EPOCHS} ({phase}) loss={epoch_loss/len(tensors):.4f}", flush=True)

    save_dir = os.path.join(MODEL_DIR, tag)
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'state_dict': net.state_dict(),
                'in_shape' : (C, H, W),
                'num_blocks': BLOCKS},
               os.path.join(save_dir, f"inn_{tag}.pth"))
    return net

# ────────── main ─────────────────────────────────────────
if __name__ == '__main__':
    DS_BOTH = load_tensors('both')   # A-모델
    DS_LH   = load_tensors('lh')     # B-LH
    DS_HL   = load_tensors('hl')     # B-HL
    DS_FULL = load_tensors('full')   # C-모델 (LL+LH+HL+HH)

    train('both',  DS_BOTH, build_inn)   # A
    train('lh',    DS_LH, build_inn)     # B1
    train('hl',    DS_HL, build_inn)     # B2
    train('full',  DS_FULL, build_inn)   # C  ← 새로 추가
