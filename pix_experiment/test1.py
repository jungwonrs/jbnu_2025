import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
import json

print("AI 평가 모델(ResNet-50) 준비중..")

model_C = models.resnet50(pretrained=True)
model_C.eval()

# ImageNet Settings
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
response = requests.get(LABELS_URL)
imagenet_labels = {int(k): v[1] for k, v in json.loads(response.text).items()}

# model processing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_image(model, image, labels):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    label = labels[top1_catid.item()]
    prob = top1_prob.item() * 100
    return f"{label} ({prob:.2f}%)"

print("\n 원본 이미지 불러오기 및 수정 작업")

IMG_URL = "https://cdn.pixabay.com/photo/2016/12/13/05/15/puppy-1903313_1280.jpg"
img_A_pil = Image.open(requests.get(IMG_URL, stream=True).raw).convert("RGB")
img_A = np.array(img_A_pil)
print("   -> 원본 이미지")

img_B_pil = img_A_pil.quantize(colors=16).convert("RGB")
img_B = np.array(img_B_pil)
print( "   ->포스트제이션으로 수정된 이미지 생성")

# AI 모델 평가
print("\n AI 모델이 두 이미지를 각각 평가")
prediction_A = evaluate_image(model_C, img_A_pil, imagenet_labels)
prediction_B = evaluate_image(model_C, img_B_pil, imagenet_labels)

print (f"  -> 원본 A 평가 결과 : {prediction_A}")
print (f"  -> 원본 B 평가 결과 : {prediction_B}")

print ("Residual 계산 및 복원 시도")

diff_D = img_A.astype(np.int16) - img_B.astype(np.int16)
print( "-> Residual 계산 완료")

restored_A_array = (img_B.astype(np.int16) + diff_D).clip(0, 255).astype(np.uint8)
restored_A_pil = Image.fromarray(restored_A_array)
print ("  -> 복원 완료")

# 검증
is_same = np.array_equal(img_A, restored_A_array)
print (" 원본과 동일? -> {is_same}")

print ("\n 시각화 시작")

diff_visual = ((diff_D - diff_D.min()) / (diff_D.max() - diff_D.min()) * 255).astype(np.uint8)

plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
plt.imshow(img_A)
plt.title(f"1 Original Image (A)\nPredicted: {prediction_A}", fontsize=12)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_B)
plt.title(f"2 Modified  Image (B)\nPredicted: {prediction_B}", fontsize=12)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(diff_visual, cmap='gray')
plt.title("3. Residual (D = A - B)\nThe 'Secret Key' for Restoration", fontsize=12)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(restored_A_pil)
plt.title(f"4. Perfectly Restored Image (B + D)\nIs it same as original? -> {is_same}", fontsize=12)
plt.axis('off')

plt.tight_layout()
plt.show()