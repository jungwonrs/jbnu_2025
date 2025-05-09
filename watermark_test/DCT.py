import cv2
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt

# 이미지 읽기 (그레이스케일, float32)
img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape
img = np.float32(img)

# 블록 크기 (8x8)
block_size = 8

# DCT 계수 맵 생성
dct_map = np.zeros_like(img)

# 블록별 DCT 적용
for i in range(0, h, block_size):
    for j in range(0, w, block_size):
        block = img[i:i+block_size, j:j+block_size]
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        dct_map[i:i+block_size, j:j+block_size] = np.log1p(abs(dct_block))  # log scale

# 저주파, 중주파, 고주파 마스크 생성 (중앙 대각선 기준 단순 구분)
low_freq_mask = np.zeros((block_size, block_size), dtype=np.uint8)
mid_freq_mask = np.zeros_like(low_freq_mask)
high_freq_mask = np.zeros_like(low_freq_mask)

for x in range(block_size):
    for y in range(block_size):
        if x + y <= 4:
            low_freq_mask[x, y] = 1
        elif x + y <= 10:
            mid_freq_mask[x, y] = 1
        else:
            high_freq_mask[x, y] = 1

# 전체 이미지 크기용 마스크 확장
low_map = np.tile(low_freq_mask, (h // block_size, w // block_size))
mid_map = np.tile(mid_freq_mask, (h // block_size, w // block_size))
high_map = np.tile(high_freq_mask, (h // block_size, w // block_size))

# 시각화
plt.figure(figsize=(12, 8))
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(222), plt.imshow(dct_map, cmap='gray'), plt.title('DCT (log scale)')
plt.subplot(223), plt.imshow(low_map, cmap='gray'), plt.title('Low Frequency Mask')
plt.subplot(224), plt.imshow(mid_map + 2 * high_map, cmap='jet'), plt.title('Mid (blue) & High (red) Frequency')
plt.tight_layout()
plt.show()
