import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# 이미지 읽기 (그레이스케일)
img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)

# Integer Wavelet Transform
coeffs = pywt.swt2(img, 'haar', level=1, start_level=0, axes=(-2, -1))
cA, (cH, cV, cD) = coeffs[0]

# 역변환
reconstructed = pywt.iswt2([(cA, (cH, cV, cD))], 'haar')

# 시각화
plt.figure(figsize=(10, 4))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(reconstructed, cmap='gray'), plt.title('Reconstructed')
plt.subplot(133), plt.imshow(abs(img - reconstructed), cmap='gray'), plt.title('Difference')
plt.show()
