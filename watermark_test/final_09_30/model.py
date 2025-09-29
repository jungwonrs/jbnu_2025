import os, random, pywt, torch, cv2, logging, sys, time, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import FrEIA.framework as Ff, FrEIA.modules as Fm
from pycocotools.coco import COCO
from tqdm import tqdm
from config import *
from math import pi, cos

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
    
    # [핵심 수정 1] 학습률 스케줄러 추가 (위치 수정 없음)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    _, wm_map = make_wm_bits(WM_LEN, WM_SEED, (H, W))
    wm_map = wm_map.to(DEVICE)
    wm_sign = (wm_map * 2 - 1) * WM_STRENGTH
    wm_ch = list(range(C)) if C == 2 else [1, 2]

    for ep in range(1, EPOCHS + 1):
        phase = 'A' if ep <= EPOCHS_A else 'B'
        epoch_loss = 0.0
        
        if phase == 'B':
            progress = (ep - EPOCHS_A) / EPOCHS_B
            BETA = BETA_MAX * (1 - cos(pi * progress)) / 2
        else:
            BETA = 0

        for step, x in enumerate(loader, 1):
            x = x.to(DEVICE)
            z, logJ = net(x)
            
            loss_z = LAM_Z * IMP_GAIN * z.pow(2).mean()
            loss_j = LAM_J * logJ.mean()
            
            loss = loss_z - loss_j
            
            if phase == 'A':
                loss += 0.05 * z[:, wm_ch].pow(2).mean()
            else: # phase == 'B'
                imp_scaled = 0.5 + 0.5 * build_importance_map(x).to(DEVICE)
                imp_scaled = imp_scaled.expand(x.size(0), 1, H, W)
                z_emb = z.clone()
                wm = wm_sign.unsqueeze(0) * imp_scaled
                
                target_signal = torch.zeros_like(z)
                for ch in wm_ch:
                    target_signal[:, ch] = wm.squeeze(1)
                    z_emb[:, ch] += wm.squeeze(1)
                
                x_stego, _ = net(z_emb, rev=True, jac=True)

                loss_distortion = F.mse_loss(x_stego, x)

                r = random.random()
                if r < 0.2:
                    a = random.uniform(0.5, 5.0)
                    x_atk = TF.gaussian_blur(x_stego, (int(a*4+1)|1), a)
                elif r < 0.4:
                    q = random.randint(10, 95)
                    x_atk = jpeg_tensor(x_stego[:, :1], q).repeat(1, C, 1, 1)
                elif r < 0.6:
                    noise_std = random.uniform(0.005, 0.1)
                    x_atk = (x_stego + torch.randn_like(x_stego) * noise_std).clamp(0,1)
                elif r < 0.8:
                    H_img, W_img = x_stego.shape[-2:]
                    scale = random.uniform(0.5, 0.95)
                    newH, newW = int(H_img*scale), int(W_img*scale)
                    resize = TF.resize(x_stego, [newH, newW])
                    x_atk = TF.resize(resize, [H_img, W_img])
                else:
                    H_img, W_img = x_stego.shape[-2:]
                    crop_scale = random.uniform(0.5, 0.95)
                    cropH, cropW = int(H_img*crop_scale), int(W_img*crop_scale)
                    cropped = TF.center_crop(x_stego, [cropH, cropW])
                    x_atk = TF.resize(cropped, [H_img, W_img])
                
                z_back, _ = net(x_atk, jac=False)
                recovered_signal = z_back - z

                # 1. 기존 Cosine Similarity 손실 (유지 또는 주석 처리)
                B, C_wm, H_wm, W_wm = recovered_signal[:, wm_ch].shape
                rec_flat = recovered_signal[:, wm_ch].reshape(B, -1)
                tgt_flat = target_signal[:, wm_ch].reshape(B, -1)
                cos_sim = F.cosine_similarity(rec_flat, tgt_flat, dim=1)
                wm_loss = (1. - cos_sim).mean()

                # ------------------- [핵심 수정: BCE 손실 추가] -------------------
                # 2. 복원된 신호에서 비트 로짓(logits) 추출
                #    watermark_experiment.py의 extract 로직과 유사
                s = recovered_signal.flatten(2).std(-1, keepdim=True).mean(1, keepdim=True).unsqueeze(-1) + 1e-6
                logits = (recovered_signal[:, wm_ch] / s) * LOGIT_COEFF

                # 3. 정답 비트맵 준비 (wm_map을 현재 배치에 맞게 확장)
                gt_bits_map = wm_map.unsqueeze(0).unsqueeze(0).repeat(BATCH, len(wm_ch), 1, 1)

                # 4. BCE 손실 계산
                loss_bce = F.binary_cross_entropy_with_logits(logits, gt_bits_map)
                # --------------------------------------------------------------------

                # 5. 최종 손실에 모든 항목 반영
                loss += (LAMBDA_DISTORTION * loss_distortion +
                         BETA * wm_loss +
                         LAMBDA_BCE * loss_bce)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            opt.step()
            
            epoch_loss += loss.item() * x.size(0)
        
        epoch_loss_avg = epoch_loss / len(tensors)
        log_beta = f"BETA={BETA:.2f}" if phase == 'B' else "BETA=0.00"
        logger.info(f"[{tag}] Ep{ep:03d}/{EPOCHS} ({phase}) | loss={epoch_loss_avg:.4f} | {log_beta}")
        
        # [핵심 수정 3] 스케줄러를 if문 밖으로 빼내어 매 에포크마다 실행되도록 합니다.
        scheduler.step(epoch_loss_avg)

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
    #DS_BOTH = load_tensors('both')
    #DS_LH   = load_tensors('lh')    
    #DS_HL   = load_tensors('hl')     
    DS_FULL = load_tensors('full')   

    #train('both',  DS_BOTH, build_inn)  
    #train('lh',    DS_LH, build_inn)     
    #train('hl',    DS_HL, build_inn)     
    train('full',  DS_FULL, build_inn)  

'''
Seed 부분을 개인키로 확장하는 아이디어 추가
=> Seed가 다를때 얼마나 워터마크를 찾아낼수 있는가 실험 방안 고안


'''