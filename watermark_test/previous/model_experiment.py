import argparse, os, random, math, json
from pathlib import Path

import numpy as np
import cv2, pywt, torch, torch.nn as nn, torch.optim as optim
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from pycocotools.coco import COCO
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import time

# DWT
def dwt_lh_hl(img_gray, wavelet="haar"):
    coeffs2 = pywt.dwt2(img_gray, wavelet)
    _, (LH, HL, _) = coeffs2
    return LH.astype(np.float32), HL.astype(np.float32)

# coco
def coco_image_road():
    data_dir = "C:\\Users\\seo\\Desktop\\watermark_test\\coco"
    ann_file = os.path.join(data_dir, "annotations\\instances_train2017.json")
    img_dir = os.path.join(data_dir, "train2017")

    #coco API call
    coco = COCO(ann_file)

    all_img_ids = coco.getImgIds()

    data = []
    for img_id in tqdm(all_img_ids, desc="Loading COCO images"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2GRAY)
        LH, HL = dwt_lh_hl(img)
        x = np.stack([LH/255.0, HL/255.0], axis=0)
        data.append(torch.from_numpy(x).float())

    return data
    
# INN (Graph INN 정의)

'''
32 -> 출력 채널 수
값이 크면 표현력이 증가하지만, 계산량과 메모리도 증가함
16, 32, 64, 128, 256으로 가능

3 -> 커널 크기
필터가 한번에 바라보는 공간 범위
크면 클수록 더 넓은 영역을 보고 연산하지만, 정밀도는 낮아질 수 있음
3, 5, 7 가능

padding은 이미지 크기를 입력과 같게 유지
padding = 1 -> 3x3 커널일 때 출력 크기 유지 
padding = 2 -> 5x5 커널일 때 출력 크기 유지

'''
def subnet_fc(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, c_out, 3, padding=1)
    )

def build_inn_model(channels, height, width):
    nodes = [Ff.InputNode(channels, height, width, name='input')]

    # invertible blocks
    for k in range (20):
        nodes.append(
            Ff.Node(nodes[-1],
                    Fm.AllInOneBlock,
                    {"subnet_constructor": subnet_fc},
                    name=f"inn_{k}")
        )

    nodes.append(Ff.OutputNode(nodes[-1], name="output"))
    return Ff.ReversibleGraphNet(nodes, verbose=False)

def train_inn(batch_size, epoches):
    device = torch.device("cpu")

    # data call
    '''
    batch_size = 8 -> 데이터를 8개씩 가져와서 학습
    shuffle = true -> 에포크마다 데이터 순서 섞음 (랜덤 학습 방지)
    '''
    dataset = coco_image_road()
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    # INN 초기화
    '''
    channels = 2 -> 입력 채널이 2개라는 의미임
    height, width -> 이미지 크기
    '''

    nll_coef = 0.0
    save_path = "inn_model.pth"

    model = build_inn_model(channels=2, height=128, width=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start_time = time.time()

    for epoch in range(1, epoches + 1):
        model.train()
        epoch_loss = 0.0
        for x in tqdm(loader, desc=f"Epoch {epoch}/{epoches}"):
            x = x.to(device)
            z, log_jac = model(x)
            loss = 0.5 * z.pow(2).mean() - log_jac.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        print(f"Epoch {epoch}: NLL = {epoch_loss / len(dataset):.4f}")
    torch.save({"state_dict": model.state_dict(), "in_shape": (2, 128, 128)}, save_path)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n총 학습 시간: {elapsed:.2f}초 ({elapsed/60:.2f}분)")

    print(f"Model saved to {save_path}")


'''
논문
사진 = 2000장
learning rate = 0.0001
250 epochs
bactch 4
20 invertible blocks
'''

batch_size = 4
epoches = 10

train_inn(batch_size, epoches)