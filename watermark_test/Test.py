import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

def dwt_process():
    original_img = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)
    
    '''
    haar (Haar wavelet): 가장 단순한 wavelet
    db1 ~ db38 (Daubechies): haar보다 부드럽고, 긴 필터 계수
    sym2 ~ sym20 (Symlets): Daubechies 개선형, 더 대칭
    coif1 ~ coif17 (Coiflets): 높은 평활성, 더 정교한 처리용
    bior1.3, bior2.2 (Biorthogonal): 신호 재구성에 강함, 영상압축에 자주 사용
    rbio1.3 (Reverse biorthogonal): Biorthogonal의 역방향
    dmey (Discrete Meyer): 부드러운 주파수 전환
    '''
    coeffs = pywt.dwt2(original_img, "haar")
    
    LL, (LH, HL, HH) = coeffs
    
    return original_img, LL, LH, HL, HH

def watermark_generation():
    #(comment: blockchain하고 엮는 방법 고민 필요)
    watermark_string = "watermark"

    binary_string = ''.join(format(ord(char), '08b') for char in watermark_string)

    length = len(binary_string)
    mid = length // 2

    first_half = binary_string[:mid]
    second_half = binary_string[mid:]

    return first_half, second_half

def watermark_processing():

    original_img, LL, LH, HL, HH = dwt_process()
    first_water, second_water = watermark_generation()

    # 워터마크용 캔버스 생성
    LH_input = np.zeros_like(LH, dtype=np.uint8)
    HL_input = np.zeros_like(HL, dtype=np.uint8)

    #캔버스에 워터마크 삽입
    '''
    1 parameter: 텍스트를 그릴 대상
    2 parameter: 삽입할 문자열
    3 parameter: 텍스트 좌상단 좌표, 왼쪽에서 10px, 세로 중간 위치 (좌표값)
    4 parameter: 폰트 종류 
    5 parameter: 폰트 크기
    6 parameter: 색 
    7 parameter: 선 두깨
    8 parameter: 안티앨리어싱 (부드럽게 그리기)
    '''
    cv2.putText(LH_input, first_water, (10, LH_input.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,), 2, cv2.LINE_AA)
    cv2.putText(HL_input, second_water, (10, HL_input.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,), 2, cv2.LINE_AA)

    # 워터마크 삽입
    alpha = 0.1 # 워터마크 강도. 작게하면 안보임 크게하면 찐하게 보임
    '''
    alpha * AA -> 강도 조절
    xx + alpha * AA -> 신호 삽입
    '''
    wm_LH = LH + alpha * LH_input
    wm_HL = HL + alpha * HL_input

    # IDWT (Inverse Discrete Wavelet Transform) - 역 이산 웨이블릿 변환
    '''
    윗 부분은 주파수로 쪼개서 워터마크를 넣었음
    해당 주파수를 다시 원래 이미지로 합치는 과정임
    '''

    coeffs_wm = LL, (wm_LH, wm_HL, HH)
    img_wm = pywt.idwt2(coeffs_wm, 'haar')
    img_wm = np.clip(img_wm, 0, 255).astype(np.uint8)

    return original_img, img_wm

def watermark_extract():
    alpha = 0.1 # 삽입하고 동일한 레벨
    

#=============================테스트=============================================
#
# 삽입 결과를 위한 시각확인
def visualize_subband_differences(original_img, watermarked_img,
                                  scale=10, wavelet="haar"):
    """
    원본·워터마크 이미지에서 LH·HL 서브밴드를 추출해
    (원본, 워터마크, 차이×scale) 2행×3열 플랏으로 표시
    """
    _, (LH_o, HL_o, _) = pywt.dwt2(original_img,      wavelet)
    _, (LH_w, HL_w, _) = pywt.dwt2(watermarked_img,   wavelet)

    imgs   = [LH_o, LH_w, (LH_w - LH_o) * scale,
              HL_o, HL_w, (HL_w - HL_o) * scale]
    titles = ["LH original", "LH watermarked", f"LH Δ×{scale}",
              "HL original", "HL watermarked", f"HL Δ×{scale}"]

    plt.figure(figsize=(12, 6))
    for i, (im, ttl) in enumerate(zip(imgs, titles), 1):
        plt.subplot(2, 3, i)
        plt.imshow(im, cmap="gray")
        plt.title(ttl, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# 호출값 받기
original_img, img_wm = watermark_processing()

# 출력
plt.figure(figsize=(15, 2))

plt.subplot(1, 2, 1)
plt.imshow(original_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_wm, cmap='gray')
plt.title('Watermarked Image')
plt.axis('off')

plt.tight_layout()
plt.show()

visualize_subband_differences(original_img, img_wm, scale=10)


