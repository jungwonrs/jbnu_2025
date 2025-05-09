import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기 (그레이스케일)
img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)

# 2D DWT 수행
coeffs2 = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coeffs2

# 결과 시각화
plt.figure(figsize=(10, 8))
plt.subplot(221), plt.imshow(cA, cmap='gray'), plt.title('Approximation')
plt.subplot(222), plt.imshow(cH, cmap='gray'), plt.title('Horizontal detail')
plt.subplot(223), plt.imshow(cV, cmap='gray'), plt.title('Vertical detail')
plt.subplot(224), plt.imshow(cD, cmap='gray'), plt.title('Diagonal detail')
plt.show()
