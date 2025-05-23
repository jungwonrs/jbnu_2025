import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

def dwt_process():
    original_img = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
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
    watermark_string = "watermark_seo1235555"

    binary_string = ''.join(format(ord(char), '08b') for char in watermark_string)

    mid = len(binary_string) // 2

    first_half = binary_string[:mid]
    second_half = binary_string[mid:]
    
    global BIT_LEN_EACH
    BIT_LEN_EACH = mid
    return first_half, second_half

def watermark_processing():

    original_img, LL, LH, HL, HH = dwt_process()
    first_water, second_water = watermark_generation()

    # 워터마크용 캔버스 생성
    LH_input = np.zeros_like(LH, dtype=float)
    HL_input = np.zeros_like(HL, dtype=float)

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

    cv2.putText(LH_input, first_water, (10, LH_input.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,), 2, cv2.LINE_AA)
    cv2.putText(HL_input, second_water, (10, HL_input.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,), 2, cv2.LINE_AA)
    '''

    y = LH_input.shape[0] // 2      # 가운데 한 줄
    x0 = 10                         # 왼쪽 여백
    for i, bit in enumerate(first_water):
        LH_input[y, x0 + i] = 1.0 * int(bit)
    for i, bit in enumerate(second_water):
        HL_input[y, x0 + i] = 1.0 * int(bit)

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
    img_wm = np.clip(img_wm, 0, 255)
    return original_img, img_wm

def watermark_extract(water_img):
   alpha = 0.1 # 삽입하고 동일한 레벨
   
   # 원본
   _, _, LH, HL, _ = dwt_process()

   # 워터마크
   _, (LH_w, HL_w, _) = pywt.dwt2(water_img, 'haar')[0:2]

   rec_LH_input = (LH_w - LH) / alpha
   rec_HL_input = (HL_w - HL) / alpha

   y = rec_LH_input.shape[0] // 2
   x0 = 10
   TH = 0.5  # 임계값; 알파·양자화 노이즈 감안
   bin_LH = ''.join('1' if rec_LH_input[y, x0 + i] > TH else '0' for i in range(BIT_LEN_EACH))
   bin_HL = ''.join('1' if rec_HL_input[y, x0 + i] > TH else '0' for i in range(BIT_LEN_EACH))

   bits = bin_LH + bin_HL                          
   chars = [chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)]
   return ''.join(chars)

   

#=============================테스트=============================================
#
# 삽입 결과를 위한 시각확인

def show_watermark_highlight(orig, wm,
                             alpha=0.1,
                             wavelet="haar",
                             thresh_ratio=0.7,
                             widen_px=3):
    """
    orig / wm : 원본·워터마크 이미지 (uint8 or float)
    alpha     : 삽입 시 쓴 값
    thresh_ratio : alpha * ratio 이상 차이면 '워터마크 픽셀'로 간주
    widen_px  : 시각화 시 몇 픽셀 두께로 굵게 보여줄지
    """
    # 1) 서브밴드 차이 구하기
    _, (LH_o, HL_o, _) = pywt.dwt2(orig.astype(np.float32), wavelet)
    _, (LH_w, HL_w, _) = pywt.dwt2(wm .astype(np.float32), wavelet)
    d_LH = LH_w - LH_o
    d_HL = HL_w - HL_o

    # 2) 워터마크 픽셀만 바이너리 마스크
    th = alpha * thresh_ratio
    mask_LH = (np.abs(d_LH) > th).astype(np.uint8)
    mask_HL = (np.abs(d_HL) > th).astype(np.uint8)

    # 3) 가운데 한 줄만 가져와서 가로로 복제 → thick line
    y = mask_LH.shape[0] // 2
    row_LH = np.tile(mask_LH[y:y+1, :], (widen_px, 1))
    row_HL = np.tile(mask_HL[y:y+1, :], (widen_px, 1))

    # 4) 컬러로 덧칠 (초록색) & 원본 서브밴드 톤다운
    def overlay(base, row_mask):
        base      = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        base_rgb  = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        row_rgb   = np.zeros_like(base_rgb)
        row_rgb[:, :, 1] = 255          # pure green
        # row 위치 맞추기
        h_off = (base.shape[0] - row_mask.shape[0]) // 2
        base_rgb[h_off:h_off+row_mask.shape[0], :, :] = np.where(
            row_mask[..., None].astype(bool),
            row_rgb[h_off:h_off+row_mask.shape[0], :, :],
            base_rgb[h_off:h_off+row_mask.shape[0], :, :]
        )
        return base_rgb

    vis_LH = overlay(LH_o, row_LH)
    vis_HL = overlay(HL_o, row_HL)

    # 5) 표시
    plt.figure(figsize=(10, 5))
    for i, (img, ttl) in enumerate([(vis_LH, "LH 서브밴드 + 워터마크"),
                                    (vis_HL, "HL 서브밴드 + 워터마크")], 1):
        plt.subplot(1, 2, i)
        plt.imshow(img); plt.title(ttl); plt.axis("off")
    plt.tight_layout(); plt.show()


# 호출값 받기
original_img, img_wm = watermark_processing()
watermark_string = watermark_extract(img_wm)
print("restore-->", repr(watermark_string))

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

show_watermark_highlight(original_img, img_wm)


