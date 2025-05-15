import numpy as np, cv2, pywt, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim, FrEIA.framework as Ff, FrEIA.modules as Fm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from pathlib import Path
from math import ceil, sqrt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device) #cuda:0 나오면 gpu 사용중

# --------1. 이미지 & 워터마크 처리------------------
# 이미지 불러오기
img_path = 'test_image.jpg'
assert Path(img_path).exists(), "이미지 파일이 없음!"

#original_image = cv2.imread(img_path)
original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
original_image = cv2.resize(original_image, (256, 256)).astype(np.float32) / 255.0

channels = cv2.split(original_image)


# 워터 마크 처리
# **** 블록체인 적용방안 고민 ******
watermark_string = "watermark_seo123456789"
binary_string = ''.join(format(ord(char), '08b') for char in watermark_string)
mid = len(binary_string) // 2
first_half = binary_string[:mid]
second_half = binary_string[mid:]

'''
side -> 비트 문자열을 정사각형 형태의 배열로 변경
padded -> 문자열의 길이에 따라 부족한 칸을 0으로 채움
'''
def str_to_arry(bit_str):
    side = ceil(np.sqrt(len(bit_str)))
    padded = bit_str.ljust(side * side, '0')
    return np.fromiter(padded, int).reshape(side, side) * 255, side

wm1, side1 = str_to_arry(first_half)
wm2, side2 = str_to_arry(second_half)

global BIT_LEN_EACH
BIT_LEN_EACH = mid

# 주파수 분리
LL,(LH,HL,HH) = pywt.dwt2(original_image, 'haar')

'''
flatten -> 다차원 배열을 1차원으로 평탄화
ex) [[1, 2],
    [3, 4]] 를 flatten 시키면
    [1, 2, 3, 4] 요렇게 됨

wm / 255.0 -> 정규화 (워터마크 값을 0~1 범위로 변경)
'''
LH_flat = LH.flatten()
HL_flat = HL.flatten()
wm1_flat = wm1.flatten() / 255.0
wm2_flat = wm2.flatten() / 255.0

# --------2. INN 셋팅 ------------------

'''
INN에 넣을 입력 벡터를 준비하는 과정
[LH 값 / HL 값 / 워터마크 앞 절반 / 워터마크 뒤 절반] 형태로 하나의 백터로 만듬
학습해야하는건 4가지 정보인데 그것을 하나의 백터 형태로 수정해서 입력
'''
x_vec = torch.tensor(
    np.hstack([LH_flat, HL_flat, wm1_flat, wm2_flat]),
    dtype=torch.float32,
    device=device
)

'''
input_dim을 통해 명시적으로 넘겨야 INN이 입력 벡터의 크기를 알 수 있음
x_vec은 그냥 텐서임. .shape[0]을 통해서 크기를 가져오지 않으면 INN에 입력불가
'''
input_dim = x_vec.shape[0]

'''
GLOWCouplingBlock: 
-> 역전 가능한 변환(INN)을 구성하는 핵심 모듈
-> 입력을 절반으로 나눠 한쪽 (x1)을 기준으로 나머지 (x2)를 변형하는 방식
-> 입력 x = [x1, x2]
   순방향: 
     x2' = x2 + f(x1)
     out = [x1, x2']
   역방향:
     x2 = x2' - f(x1)
f(x1)을 계산하는게 subnet

din:
-> subnet에 입력되는 벡터의 크기
-> 자동으로 설정되며, 보통은 전체 입력 벡터의 절반
-> x_vec이 2048차원이면 subnet에는 din=1024, dout=1024가 들어감
-> 커지면 더 많은 정보를 처리할 수 있고, 복잡한 구조 학습이 가능
   연산량 증가, 과적합 위험, 학습 속도 느리다는 단점 존재

nn.Linear(din, 128):
-> 128은 중간층의 뉴런 수 
-> 자유롭게 설정이 가능함

hidden_size = int(din * 1.0):
-> 중간층(은닉층)크기를 동적으로 설정
-> 0.5, 2.0 등의 비율로 조정가능
'''
def subnet(din, dout):
    print("din>>>>>>>>", din)
    hidden_size = min(4096, int(din //2))
    return nn.Sequential(
        nn.Linear(din, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, dout)
    )

'''
그래프 구조를 정의할 때 가장 첫 번째 입력 노드를 만드는 단계

Ff.InputNode:
-> FrEIA 프레임워크에서 사용하는 입력 노드 생성함수
-> input_dim: 입력 벡터의 차원 수 
-> name: 노드의 이름 (그래프 내에서 추적용)

'''
nodes = [Ff.InputNode(input_dim, name='in')]

nodes.append(
    Ff.Node(nodes[-1],
            Fm.PermuteRandom,
            {'seed': 0},
            name='perm0')
)

'''
GLOWCouplingBlock을 그래프에 추가하는 코드
입력 벡터를 반으로 나눈 뒤, 그 중 절반을 기준으로 나머지를 subnet을 이용해 변형해라

nodes[-1]:
-> 이전 노드를 가리킴

Fm.GLOWCouplingBlock:
-> 사용할 모듈 

{'subnet_constructor': subnet, 'clamp': 1.0}:
-> 파라미터

name='cb':
-> 디버깅, 그래프 추적용

num_block:
-> 블록 갯수
-> 많아지면 학습이 강해짐
-> 1~2개 간단한 실험, 3~5개 고해상도
'''

num_block = 4

for i in range(num_block):
    nodes.append(
        Ff.Node(
            nodes[-1],
            Fm.GLOWCouplingBlock,
            {
                'subnet_constructor': subnet,
                'clamp': 0.5
            },
            name = f'cb{i}'
        )
    )

'''
출력을 명시적으로 지정해주기 위한 코드
'''
nodes.append(Ff.OutputNode(nodes[-1], name='out'))

'''
만든 nodes 리스트를 기반으로 
하나의 ReversibleGraphNet(가역 신경망 모델) 객체를 생성
'''
inn = Ff.ReversibleGraphNet(nodes).to(device)

for m in inn.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.1)  # 평균 0, 분산 유지
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# --------3. 학습 ------------------

'''
**** 강건성을 확보하기 위한 튜닝 필수 ******
**** MSE + BCE 조합??? ******

같은 입력을 n개로 복사한 가짜 배치를 반들어 학습에 사용
INN이 자기 자신을 재구성하는 방식으로 학습되기 때문에 다양한 데이터가 없어도 자기복제 학습이 가능

x_vec:
-> 지금까지 만든 하나의 입력 벡터

unsqueeze(0):
-> 차원을 늘림
-> 배치 차원 추가이며, 0번째 차원을 새로 추가한다는 뜻

repeat(64):
-> 같은 벡터를 64번 복사해서 배치(batch 생성)
-> 32: 더 빠른 학습, 다양성은 낮아짐
   64: 적당..?
   128: 추정치 안정됨, 메모리 시간 부담
   256: 학습 더 안정화
'''
noise_std = 5e-4
x_noisy = x_vec + noise_std * torch.randn_like(x_vec)
batch = x_noisy.unsqueeze(0).repeat(32, 1)

'''
pyTorch의 Adam 최적화기 설정

inn.parameters(): 
-> INN 모델 안의 모든 학습 가능한 파라미터들

1e-3:
-> 학습률
-> 1e-3이 0.001로 가장 많이 쓰이는 기본값
-> 너무 크면 발산 위험, 너무 작으면 수렴이 안됨
-> 발산: 손실이 오히려 커지거나 NAN(무한대)가 발생
        loss 값이 수시로 오르내림 -> 발산 가능성
-> 수렴: 학습이 진행될 수록 (loss)이 점점 줄고, 일정 수준에서 안정됨
        PSNR/SSIM이 전혀 안올라감 -> 무의미한 학습
        
-> 보통 범위가 (0.0001~0.01)
-> 1e-3 -> 1e-2 -> 1e-4 순서대로 하나씩 해보면됨

nn.MSELoss():
-> Mean Squared Error (평균제곱오차)
-> INN은 입력 -> 출력 -> 역방향 복원까지 하니까 
   입력 vs 복원 결과의 차이를 최소화하려고 MSE를 사용

'''
opt,loss_fn = optim.Adam(inn.parameters(),1e-3), nn.MSELoss()

'''
같은 입력을 넣고 -> INN이 그것을 바꾸고 -> 다시 되돌릴 수 있도록 학습하는 과정

for ep in range(300):
-> ep 는 epoch(학습 반복 횟수)를 의미
-> 총 300번 반복 해서 훈련

out = inn(batch)[0]:
-> INN에 batch 데이터를 넣음
-> out: 워터마크가 삽입된(변형된) 데이터
-> [0]: ReversibleGraphNet 출력에 의해 튜플이라 첫 번째 요소만 꺼냄

recon = inn(out, rev=True)[0]:
-> rev = True: INN의 역방향 실행
   out을 넣으면 원래 batch가 복원되도록 학습하는것이 목표
-> recon: 복원된 데이터

loss = loss_fn(recon, batch):
-> 복원된 recon과 원래 입력 batch 사이의 오차 계산

opt.zero_grad(); loss.backward()l opt.step():
-> zero_grad(): 이전 반복에서 남은 gradient 초기화
-> backward(): 손실에 대한 gradient 계산 (역전파)
-> step() 파라미터 업데티 (gradient descent)

if ep%100==0: print(f'epoch {ep:3d}  loss={loss.item():.3e}'):
-> 매 100 epoch마다 현재 손실값 출력
-> loss.item()은 PyTorch 텐서를 일반 숫자로 변환

'''

bce = nn.BCEWithLogitsLoss(reduction='mean')
y   = 0.05
opt = optim.Adam(inn.parameters(), lr=1e-4)

LH_size = LH.size
HL_size = HL.size
wm_len  = wm1.size

batch_sz = batch.shape[0]        

wm_target = torch.tensor(wm1_flat[:wm_len],   
                         dtype=torch.float32,
                         device=device).unsqueeze(0).repeat(batch_sz, 1)

for ep in range(1):
    out   = inn(batch)[0]
    recon = inn(out, rev=True)[0]

    pred_bits = recon[:, LH_size+HL_size : LH_size+HL_size+wm_len]
    
    loss_img  = loss_fn(recon, batch)
    loss_bit  = bce(pred_bits, wm_target)
    loss = loss_img + y*loss_bit
    opt.zero_grad()
    loss.backward() 
    torch.nn.utils.clip_grad_norm_(inn.parameters(), 1.0)
    opt.step()
    if ep%100==0: 
        print(f'epoch {ep:3d}  loss={loss.item():.3e}')

# --------4. 인코딩 ------------------
'''
INN 학습이 끝난 후
워터마크를 실제로 이미지에 삽입하는 단계

학습된 INN에 입력 -> 출력 결과는
[LH, HL, wm1, wm2]  순으로 정렬된 벡터 그 중에서 변형된 LH와 HL만 추출해서
LL과 HH는 원본 그대로 사용

LH_size= LH.size, HL_size = HL.size:
-> 원본 이미지에서 나온 LH, HL 고주파 성분의 크기 저장

LH_stego = out[0, :LH_size]:
-> INN의 출력 중 앞부분은 LH 성분
-> detach(): 학습 그래프에서 분리 (추론 시 필요)
-> view(LH.shape): 다시 2D 배열로 복원
-> .numpy(): PyTorch 텐서를 Numpy 배열로 변환
-> 즉, 워터마크 삽입 후 바뀐 LH 성분 복원

HL_stego = out[0, LH_size:LH_size+HL_size]:
-> LH 뒤로 이어지는 HL 성분
-> 워터마크 삽입 후 바뀐 HL 성분 복원

stego = pywt.idwt2((LL, (LH_stego, HL_stego, HH)), 'haar'):
-> 워터마크 삽입된 stego 이미지 생성

stego = np.clip(stego, 0, 1):
-> 이미지 재조합 과정에서 0보다 작거나 1보다 큰 값이 나올 수 있음
-> clip()으로 0~1 범위에 맞춰서 이미지 안정화 

'''
#LH_size= LH.size
#HL_size = HL.size

LH_stego = out[0, :LH_size].detach().cpu().view(LH.shape).numpy()
HL_stego = out[0, LH_size:LH_size+HL_size].detach().cpu().view(HL.shape).numpy()

stego = pywt.idwt2((LL, (LH_stego, HL_stego, HH)), 'haar')
stego = np.clip(stego, 0, 1)

# --------5. 추출 ------------------
'''
INN 출력 (x_back)에서 워터마크 비트를 추출하고 문자열로 복원

x_back = inn(out.detac(), rev=True)[0]:
-> out: INN 순방향 출력값
-> .detach(): 그래프 끊기
-> rev=True: INN 역방향 실행

wm1_bits = (x_back[0, LH_size + HL_size : LH_size + HL_size + wm_len] > 0.5).int().cpu().numpy(), wm2_bits = (x_back[0, -wm_len:] > 0.5).int().cpu().numpy():
-> 워터마크를 위치를 가져옴

wm_all_bits = np.concatenate([wm1_bits, wm2_bits]):
-> 원래 문자열로 만드는것

wm_rec = wm_all_bits.reshape(int(np.sqrt(len(wm_all_bits))), -1) * 255:
-> 시각화용 배열 만들기
'''

def decode_watermark(bits):
    chars = [chr(int(''.join(str(int(b)) for b in bits[i:i+8]), 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars).rstrip('\x00')

x_back = inn(out.detach(), rev=True)[0]

#wm_len = wm1.size

wm1_bits = (x_back[0, LH_size + HL_size : LH_size + HL_size + wm_len] > 0.5).int().cpu().numpy()
wm2_bits = (x_back[0, -wm_len:] > 0.5).int().cpu().numpy()
wm_all_bits = np.concatenate([wm1_bits, wm2_bits])

side = ceil(np.sqrt(len(wm_all_bits)))
wm_rec = np.pad(wm_all_bits, (0, side*side - len(wm_all_bits))).reshape(side, side) * 255

wm_raw_bits = wm_all_bits[:len(binary_string)]
print("Recovered watermark:", decode_watermark(wm_raw_bits))

# --------6. 품질 수치 및 시각화 ------------------
'''
PSNR:
-> Peak Signal-to-Noise Ratio
-> 원본과 변형 이미지 간의 차이를 신호 대비 노이즈로 해석
-> 높을수록 품질이 좋다는 걸 나타내는 수치 (단위: dB)

SSIM:
-> Structural Similarity Index
-> 이미지의 구조적 유사성을 측정
-> 밝기, 대비, 구조를 함께 고려
-> 0~1 사이의 값 이며, 1이면 완전히 동일


'''

print("PSNR:", psnr(original_image, stego))
print("SSIM:", ssim(original_image, stego, data_range=1.0))
plt.figure(figsize=(12,4))
def show(idx, title, img, vmin=0, vmax=1):
    plt.subplot(1,5,idx)
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis('off')

show(1, 'Original', original_image)
show(2, 'Stego',    stego)

'''
어디가 바뀌였는지 표현
바뀐 부분은 밝게 보임
안바뀐 부분은 검정색에 가까움
시각적 손상 영역 확인
워터마크가 주로 어디에 퍼졌는지 추정
압축/블러 후에 차이 확산되는지 비교가능
'''
show(3, '|diff|×5', np.abs(original_image - stego) * 5)

wm_in_vis = np.concatenate([wm1, wm2], axis=1)  
show(4, 'WM in (merged)', wm_in_vis, vmin=0, vmax=255)
show(5, 'WM out (merged)', wm_rec, vmin=0, vmax=255)

plt.tight_layout()
plt.show()