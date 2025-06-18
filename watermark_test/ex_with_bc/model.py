import os, random, pywt, torch, cv2, logging, sys, time, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import FrEIA.framework as Ff, FrEIA.modules as Fm
from pycocotools.coco import COCO
from tqdm import tqdm
from config import *

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

# 학습을 위한 비트 임베딩 및 이진화 연산을 통해 학습
class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(_, x):
        return x.sign()
    @staticmethod
    def backward(_, g):
        return g          
sign_ste = SignSTE.apply

# Model 학습
def train(tag, tensors, build_inn):
    global IMP_W
    IMP_W = None

    t_start = time.perf_counter()

    loader = torch.utils.data.DataLoader(
        tensors,
        batch_size = BATCH,
        shuffle = True,
        drop_last = True
    )

    C, H, W = tensors.shape[1:]
    net = build_inn(C, H, W).to(DEVICE)
    
    # beta1 = 0.5 (기본값: 0.9, moving average of gradient의 지수 감쇠율), 
    # beta2 = 0.999 (기본값: 0.999, gradient 제곱의 moving average의 지수 감쇠율)
    # gradient 설정하기 위해서 만지는 값
    opt= optim.Adam(net.parameters(), lr=LR, betas=(0.5, 0.999))

    # 랜덤 워터마크 생성
    _, wm_map = make_wm_bits(WM_LEN, WM_SEED, (H, W))

    # 워터마크 강도 설정
    wm_sign = (wm_map * 2 - 1).to(DEVICE) * WM_STRENGTH

    # 워터마크를 어디다 넣을지 지정하는 리스트
    wm_ch = list(range(C)) if C == 2 else [1, 2]

    for ep in range(1, EPOCHS +1):
        phase = 'A' if ep <= EPOCHS_A else 'B'
        epoch_loss = 0.0
        for step, x in enumerate(loader, 1):
            x = x.to(DEVICE)
            z, logJ = net(x)
            # 기본 손실 값 (중요)
            loss = LAM_Z * IMP_GAIN * z.pow(2).mean() - LAM_J * logJ.mean()
            
            if phase == 'A':
                # Loss A 손실 값 (중요)
                # Loss A = 기본 손실 값 + 0.05 * z[:, wm_ch].pow(2).mean()
                # 워터마킹이 쉽고 안정적으로 심어지도록
                loss += 0.05 * z[:, wm_ch].pow(2).mean()
                if IMP_W is None and step == 1:
                    IMP_W = build_importance_map(x)

            else:
                z_emb = z.clone()
                for ch in wm_ch:
                    imp_scaled = (0.5 + 0.5 * IMP_W.to(DEVICE))
                    wm = wm_sign.unsqueeze(0) * imp_scaled
                    z_emb[:, ch] += wm.squeeze(1)
                
                x_stego, _ = net(z_emb, rev=True, jac=True)
                # Loss B 손실 값 (중요)
                # Loss B = 기본 손실 값 + LAMBDA_CLEAN * (x_stego - x).abs().mean() + beta * wm_loss
                # 워터마크 이미지 임베딩, 복원력 보장
                clean_loss = LAMBDA_CLEAN * (x_stego - x).abs().mean()
                loss += clean_loss

                # 워터마크 강인성을 위한 공격 학습
                # 1/5씩 확률로 공격을 고르고 랜덤값을 0~1 범위에서 5등분 진행
                r = random.random()

                # Gaussian Blur
                if r < 0.2:
                    a = random.uniform(0.5, 5.0)
                    x_atk = TF.gaussian_blur(x_stego, (int(a*4+1)|1), a)
                
                # JPEG 압축
                elif r < 0.4:
                    q = random.randint(10, 95)
                    x_atk = jpeg_tensor(x_stego[:, :1], q).repeat(1, C, 1, 1)
                
                # Gaussain Noise
                elif r < 0.6:
                    noise_std = random.uniform(0.005, 0.1)
                    x_atk = (x_stego + torch.randn_like(x_stego) * noise_std).clamp(0,1)

                # Resize 
                elif r < 0.8:
                    H, W = x_stego.shape[-2:]
                    scale = random.uniform(0.5, 0.95)
                    newH, newW = int(H*scale), int(W*scale)
                    resize = TF.resize(x_stego, [newH, newW])
                    x_atk = TF.resize(resize, [H, W])

                # Center Crop
                else:
                    H, W = x_stego.shape[-2:]
                    crop_scale = random.uniform(0.5, 0.95)
                    cropH, cropW = int(H*crop_scale), int(W*crop_scale)
                    cropped = TF.center_crop(x_stego, [cropH, cropW])
                    x_atk = TF.resize(cropped, [H, W])
                
                z_back, _ = net(x_atk, jac = False)
                scale = SCALE_LOGIT / WM_STRENGTH
                logits = torch.stack([z_back[:, ch]*scale for ch in wm_ch], 1)
                target = (((wm_sign.sign().unsqueeze(0).unsqueeze(1))+1)*0.5).expand_as(logits)
                wm_loss = F.binary_cross_entropy_with_logits(logits, target)

                # 최종 Phase B loss 값 결정 (중요)
                loss += BETA * wm_loss

            # gradient 초기화 -> gradient 계산 -> gradient clipping -> 파라미터 업데이트 -> epoch별 loss 집계
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item() * x.size(0)

        logger.info(f"[{tag}] Ep{ep:03d}/{EPOCHS} ({phase})"f"loss={epoch_loss/len(tensors):.4f}")

        #print(f"[{tag}] Ep{ep:03d}/{EPOCHS} ({phase}) loss={epoch_loss/len(tensors):.4f}", flush=True)

    save_dir = os.path.join(MODEL_DIR, tag)
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'state_dict': net.state_dict(),
                'in_shape' : (C, H, W),
                'num_blocks': BLOCKS},
               os.path.join(save_dir, f"inn_{tag}.pth"))
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t_start
    logger.info(f"[{tag}] Training finished in {elapsed/60:.2f} min " f"({elapsed:.1f} s)")

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
    bs = x_bchw.size(0)
    buf = []
    for k in range(bs):
        u8 = (x_bchw[k, 0].clamp(0, 1)*255).detach().cpu().numpy().astype(np.uint8)
        enc = cv2.imencode('.jpg', u8, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        dec = cv2.imdecode(enc, 0).astype(np.float32) / 255.0
        buf.append(torch.from_numpy(dec))
    return torch.stack(buf, 0).unsqueeze(1).to(x_bchw.device)


# ────────── 실행 ──────────

if __name__ == '__main__':
    DS_BOTH = load_tensors('both')
    DS_LH   = load_tensors('lh')     # B-LH
    DS_HL   = load_tensors('hl')     # B-HL
    DS_FULL = load_tensors('full')   # C-모델 (LL+LH+HL+HH)

    train('both',  DS_BOTH, build_inn)   # A
    train('lh',    DS_LH, build_inn)     # B1
    train('hl',    DS_HL, build_inn)     # B2
    train('full',  DS_FULL, build_inn)   # C

'''
Seed 부분을 개인키로 확장하는 아이디어 추가
=> Seed가 다를때 얼마나 워터마크를 찾아낼수 있는가 실험 방안 고안


'''