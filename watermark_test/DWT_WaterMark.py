import cv2
import numpy as np
import pywt

# ------------------------------
# 유틸: 문자열 → 비트 배열
def str2bits(s: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(s.encode(), dtype=np.uint8))

# 유틸: 비트 배열 → 문자열
def bits2str(bits: np.ndarray) -> str:
    bytes_ = np.packbits(bits.astype(np.uint8))
    return bytes_.tobytes().decode(errors="ignore")

# ------------------------------
def embed_dwt_watermark(img_gray: np.ndarray,
                        wm_bits: np.ndarray,
                        alpha: float = 6.0,
                        wavelet: str = "haar") -> np.ndarray:
    """
    1–2레벨 HL/LH 서브밴드에 워터마크 비트를 분산 삽입
    """
    if img_gray.dtype != np.float32:
        img_gray = np.float32(img_gray)

    # DWT 1, 2 레벨
    coeffs_lvl1 = pywt.dwt2(img_gray, wavelet)   # level‑1
    LL1, (LH1, HL1, HH1) = coeffs_lvl1
    coeffs_lvl2 = pywt.dwt2(LL1, wavelet)        # level‑2 (LL1 안에서)
    LL2, (LH2, HL2, HH2) = coeffs_lvl2

    # 두 레벨 HL/LH 계수 펼침
    carriers = [LH1.flatten(), HL1.flatten(),
                LH2.flatten(), HL2.flatten()]
    carrier_len = sum(len(c) for c in carriers)

    # 워터마크 비트를 반복해 전체 길이에 맞춤
    wm_long = np.resize(wm_bits, carrier_len)

    # 스프레드 스펙트럼 삽입: 계수에 ±alpha 추가
    inserted = []
    start = 0
    for c in carriers:
        seg = wm_long[start:start + len(c)]
        inserted.append(c + alpha * (2 * seg - 1))  # 0→‑alpha, 1→+alpha
        start += len(c)

    # 계수 복원
    LH1_ = inserted[0].reshape(LH1.shape)
    HL1_ = inserted[1].reshape(HL1.shape)
    LH2_ = inserted[2].reshape(LH2.shape)
    HL2_ = inserted[3].reshape(HL2.shape)

    # 역 DWT
    LL1_ = pywt.idwt2((LL2, (LH2_, HL2_, HH2)), wavelet)
    watermarked = pywt.idwt2((LL1_, (LH1_, HL1_, HH1)), wavelet)

    # 값 범위 정리
    return np.uint8(np.clip(watermarked, 0, 255))

# ------------------------------
def extract_dwt_watermark(img_gray_wm: np.ndarray,
                          original_img_gray: np.ndarray,
                          wm_bits_len: int,
                          alpha: float = 6.0,
                          wavelet: str = "haar") -> np.ndarray:
    """
    원본 대비 차이를 이용해 워터마크 비트 복원
    (블라인드 추출이 아니므로 원본 필요)
    """
    # 2‑레벨 DWT 모두 수행
    def dwt_levels(img):
        c1 = pywt.dwt2(np.float32(img), wavelet)
        LL1, (LH1, HL1, HH1) = c1
        c2 = pywt.dwt2(LL1, wavelet)
        LL2, (LH2, HL2, HH2) = c2
        return [LH1.flatten(), HL1.flatten(),
                LH2.flatten(), HL2.flatten()]

    carriers_wm = dwt_levels(img_gray_wm)
    carriers_orig = dwt_levels(original_img_gray)

    # 차이 → 부호 → 비트
    diff = np.concatenate([w - o for w, o in zip(carriers_wm, carriers_orig)])
    recovered = (diff >= 0).astype(np.uint8)[:wm_bits_len]  # sign test

    return recovered

# ------------------------------
if __name__ == "__main__":
    # ❶ 그림 불러오기
    cover = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)

    # ❷ 워터마크 메시지 준비
    message = "© 2025 MyCompany"
    wm_bits = str2bits(message)

    # ❸ 삽입
    watermarked = embed_dwt_watermark(cover, wm_bits, alpha=6)

    cv2.imwrite("watermarked.png", watermarked)

    # ❹ 추출 (원본 필요 – 비블라인드 버전)
    recovered_bits = extract_dwt_watermark(watermarked, cover, len(wm_bits))
    recovered_msg = bits2str(recovered_bits)

    print("복원 메시지:", recovered_msg)
