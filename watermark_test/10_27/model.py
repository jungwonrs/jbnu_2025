import os, random, pywt, torch, cv2, logging, sys, time, numpy as np
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import FrEIA.framework as Ff, FrEIA.modules as Fm
from pycocotools.coco import COCO
from tqdm import tqdm
from config import *
from math import pi, cos
from torch.optim.lr_scheduler import CosineAnnealingLR


IMP_W = None

logging.basicConfig(
    level=logging.INFO,                                 
    format="%(asctime)s %(message)s",                   
    handlers=[
        logging.FileHandler("log.txt", mode="a", encoding="utf-8"),  
        logging.StreamHandler(sys.stdout)                             
    ]
)
logger = logging.getLogger(__name__)

# ────────── INN 정의 ──────────

# 은닉 채널 32로 정의, 커널 크기는 3x3
def subnet(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, 32, 3, padding=1),
        nn.ReLU(0),
        nn.Conv2d(32, c_out, 3, padding=1)
    )

def build_inn(C, H, W):
    nodes = [Ff.InputNode(C, H, W, name='in')]
    if C == 1:
        nodes.append(Ff.Node(nodes[-1],
                      Fm.HaarDownsampling,
                      {"rebalance": False,
                       "order_by_wavelet" : True},
                      name='haar'))
    for k in range(BLOCKS):
        nodes.append(
            Ff.Node(
                nodes[-1],
                Fm.AllInOneBlock,
                {"subnet_constructor": subnet},
                name=f'inn_{k}' ))
    
    nodes.append(Ff.OutputNode(
        nodes[-1],
        name = 'out'
    ))

    return Ff.ReversibleGraphNet(nodes, verbose=False)

# ────────── INN 학습 ──────────

# Model 학습
def train(tag, tensors, build_inn):
    t_start = time.perf_counter()

    loader = torch.utils.data.DataLoader(
        tensors,
        batch_size=BATCH,
        shuffle=True,
        drop_last=True
    )

    C, H, W = tensors.shape[1:]
    net = build_inn(C, H, W).to(DEVICE)
    
    opt = optim.Adam(net.parameters(), lr=LR, betas=(0.5, 0.999))
    
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1)

    _, wm_map = make_wm_bits(WM_LEN, WM_SEED, (H, W))
    wm_sign = (wm_map * 2 - 1).to(DEVICE) * WM_STRENGTH
    wm_ch = list(range(C)) if C == 2 else [1, 2]

    for ep in range(1, EPOCHS + 1):
        phase = 'A' if ep <= EPOCHS_A else 'B'
        epoch_loss = 0.0
        
        for step, x in enumerate(loader, 1):
            x = x.to(DEVICE)
            z, logJ = net(x)
            
            loss_z = LAM_Z * IMP_GAIN * z.pow(2).mean()
            loss_j = LAM_J * logJ.mean()
            
            loss = loss_z - loss_j
            
            if phase == 'A':
                loss += 0.05 * z[:, wm_ch].pow(2).mean()
            else: # phase == 'B'
                # ================================================================= #
                # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분이 수정되었습니다 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
                # ================================================================= #
                
                # 진행도: [0,1] 범위로 안전하게 클램핑
                progress = (ep - EPOCHS_A) / max(1, EPOCHS_B)
                progress = max(0.0, min(1.0, progress))

                BETA = BETA_MAX * (1 - math.cos(math.pi * progress)) / 2

                # --- Stego 이미지 생성 ---
                imp_scaled = 0.5 + 0.5 * build_importance_map(x).to(DEVICE)
                imp_scaled = imp_scaled.expand(x.size(0), 1, H, W)
                z_emb = z.clone()
                wm = wm_sign.unsqueeze(0) * imp_scaled
                
                target_signal = torch.zeros_like(z)
                for ch in wm_ch:
                    target_signal[:, ch] = wm.squeeze(1)
                    z_emb[:, ch] += wm.squeeze(1)
                
                x_stego, _ = net(z_emb, rev=True, jac=False) # jac=False로 속도 향상
                loss_distortion = F.mse_loss(x_stego, x)

                # --- 공격 커리큘럼 시작 ---
                x_atk = x_stego.clone()

                # 10% 확률로 공격 없이 클린 이미지를 학습
                if random.random() < 0.10:
                    pass
                else:
                    # 공격 적용 확률을 훈련 진행도에 따라 점진적으로 증가
                    p_jpeg = 0.5 + 0.2 * progress
                    p_blur = 0.3 + 0.2 * progress
                    p_noise = 0.3 + 0.2 * progress

                    # 1) JPEG 공격
                    if random.random() < p_jpeg:
                        if progress < 0.6 or random.random() < 0.8:
                            q = random.randint(70, 95)
                        else:
                            q = random.randint(50, 69)
                        x_atk = jpeg_tensor(x_atk, q)

                    # 2) 블러(Blur) 공격
                    if random.random() < p_blur:
                        sigma_max = 1.0 + 1.0 * progress
                        a = random.uniform(0.5, sigma_max)
                        if progress > 0.7 and random.random() < 0.10:
                            a = random.uniform(2.0, 3.0)
                        k = int(a * 4 + 1) | 1
                        x_atk = TF.gaussian_blur(x_atk, (k, k), a)

                    # 3) 노이즈(Noise) 공격
                    if random.random() < p_noise:
                        std_max = 0.03 + 0.02 * progress
                        noise_std = random.uniform(0.1, std_max)
                        if progress > 0.7 and random.random() < 0.10:
                            noise_std = random.uniform(0.05, 0.08)
                        x_atk = (x_atk + torch.randn_like(x_atk) * noise_std).clamp(0, 1)

                z_back, _ = net(x_atk, jac=False)
                recovered_signal = z_back - z
                
                # ================================================================= #
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 여기까지가 수정된 부분입니다 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
                # ================================================================= #
                
                B, C_wm, H_wm, W_wm = recovered_signal[:, wm_ch].shape
                rec_flat = recovered_signal[:, wm_ch].reshape(B, -1)
                tgt_flat = target_signal[:, wm_ch].reshape(B, -1)
                
                cos_sim = F.cosine_similarity(rec_flat, tgt_flat, dim=1)
                wm_loss_cos = (1. - cos_sim).mean()

                wm_loss_mag = F.l1_loss(rec_flat, tgt_flat)
                
                #mag_weight = 0.2
                wm_loss = wm_loss_cos + MAG_WEIGHT * wm_loss_mag
                
                loss += LAMBDA_DISTORTION * loss_distortion + BETA * wm_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            opt.step()
            
            epoch_loss += loss.item() * x.size(0)
        
        epoch_loss_avg = epoch_loss / len(tensors)
        log_beta = f"BETA={BETA:.2f}" if phase == 'B' else "BETA=0.00"
        logger.info(f"[{tag}] Ep{ep:03d}/{EPOCHS} ({phase}) | loss={epoch_loss_avg:.4f} | {log_beta}")
        
        scheduler.step()

    save_dir = os.path.join(MODEL_DIR, tag)
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'state_dict': net.state_dict(),
                'in_shape' : (C, H, W),
                'num_blocks': BLOCKS},
               os.path.join(save_dir, f"inn_{tag}.pth"))
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t_start
    logger.info(f"[{tag}] Training finished in {elapsed/60:.2f} min ({elapsed:.1f} s)")

    return net

# ────────── 이미지 처리 함수들 ──────────
def load_tensors(split):
    coco = COCO(TRAIN_JSON)
    ids = random.sample(coco.getImgIds(), N_IMG)
    data = []
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

def dwt_coeffs(gray):
    LL, (LH, HL, HH) = pywt.dwt2(gray, WAVELET)
    return [c.astype(np.float32) for c in (LL, LH, HL, HH)]

# ────────── 유틸리티 함수들 ──────────
def make_wm_bits (len_bits, seed, shape):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=len_bits, dtype=np.uint8)
    h, w = shape
    flat = np.repeat(bits, (h*w + len_bits - 1)//len_bits)[:h*w]
    flat = flat.reshape(h,w)
    return bits, torch.from_numpy(flat).float()

# build_importance_map, sobel_kernel은
# 사람이 잘 보는 부분/잘 안 보는 부분을 "수치적으로" 구별해서, 워터마크를 “덜 티나게” 심거나, "효과적으로" 임베딩
@torch.no_grad()
def build_importance_map(batch, k=3, eps=1e-6):
    lum = batch[:,1:3].mean(1, keepdim=True) if batch.size(1)>=3 else batch.mean(1, keepdim=True)
    kx, ky = sobel_kernel(k)
    kx, ky = kx.to(batch), ky.to(batch)
    gx = F.conv2d(lum, kx, padding=k//2)
    gy = F.conv2d(lum, ky, padding=k//2)
    grad = torch.sqrt(gx.pow(2)+gy.pow(2)+eps)
    imp = 1. - grad / (grad.amax(dim=(-3, -2, -1), keepdim=True)+eps)
    return imp.mean(0, keepdim=True)

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

# JPEG 압축 공격에 사용
def jpeg_tensor(x_bchw, q):
    # x_bchw: (B, C, H, W), 값 범위 [0,1]
    bs, C, H, W = x_bchw.shape
    out = torch.empty_like(x_bchw)
    for b in range(bs):
        for ch in range(C):
            u8 = (x_bchw[b, ch].clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
            enc = cv2.imencode('.jpg', u8, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
            dec = cv2.imdecode(enc, 0).astype(np.float32) / 255.0
            out[b, ch] = torch.from_numpy(dec)
    return out.to(x_bchw.device)


# ────────── 실행 ──────────

if __name__ == '__main__':
    DS_BOTH = load_tensors('both')
    DS_LH   = load_tensors('lh')    
    DS_HL   = load_tensors('hl')     
    DS_FULL = load_tensors('full')   

    train('both',  DS_BOTH, build_inn)  
    train('lh',    DS_LH, build_inn)     
    train('hl',    DS_HL, build_inn)     
    train('full',  DS_FULL, build_inn)  
