import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import requests
import json
import matplotlib.pyplot as plt

print(" ResNet-50 model Setting..")

model = models.resnet50(pretrained=True)
model.eval()

LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
response = requests.get(LABELS_URL)
imagenet_labels = {int(k): v[1] for k, v in json.loads(response.text).items()}

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def tensor_to_image_array(tensor):
    img_array = tensor.clone().detach().numpy().squeeze()
    img_array = img_array.transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = std * img_array + mean
    img_array = np.clip(img_array, 0, 1)
    return (img_array * 255).astype(np.uint8)

def evaluate_image(image_tensor, model, labels):
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top1_prob, top1_catied = torch.topk(probabilities, 1)
    label = labels[top1_catied.item()]
    prob = top1_prob.item() * 100
    return f"{label} ({prob:.2f}%)"

# image setting
print ("call image")
IMG_URL = "https://cdn.pixabay.com/photo/2016/12/13/05/15/puppy-1903313_1280.jpg"
img_A_pil = Image.open(requests.get(IMG_URL, stream=True).raw).convert("RGB").resize((224, 224))
img_A_tensor = preprocess(img_A_pil)
print ("원본 이미지 호출 성공")
original_prediction = evaluate_image(img_A_tensor, model, imagenet_labels)
print(f" 원본(A) 평가 결과: {original_prediction}")

# 적대적 키 생성
print("키 생성")
img_A_tensor.requires_grad =True

output = model(img_A_tensor.unsqueeze(0))
loss = nn.CrossEntropyLoss()
target = torch.tensor([9])
err = loss(output, target)
model.zero_grad()
err.backward()

epsilon = 0.05
grad_sign = img_A_tensor.grad.data.sign()
adversarial_key_K = epsilon * grad_sign
print (f" 키 생성. (Epsilon: {epsilon})")

print ("변조 이미지 생성")
img_B_tensor = img_A_tensor + adversarial_key_K
img_B_tensor = torch.clamp(img_B_tensor, -2.12, 2.25)
print ("변조 이미지 생성 완료")

modified_prediction = evaluate_image(img_B_tensor, model, imagenet_labels)
print (f"변조된 이미지(B) 평가 결과: {modified_prediction}")

print ("소유자 확인")

restored_A_tensor = img_B_tensor - adversarial_key_K
print ("변도된 이미지에서 키를 통해서 복원 완료")

restored_prediction = evaluate_image(restored_A_tensor, model, imagenet_labels)
print (f" -> 복원된 평가 결과: {restored_prediction}")

img_A_array = tensor_to_image_array(img_A_tensor)
restored_A_array = tensor_to_image_array(restored_A_tensor)
is_same = np.allclose(img_A_array, restored_A_array, atol=1)
print (f"동일함?   {is_same}")

print("\n✅ 6. 모든 과정을 시각화하여 보여줍니다.")
img_B_array = tensor_to_image_array(img_B_tensor)
key_K_visual_array = tensor_to_image_array(adversarial_key_K * (1/epsilon)) # 시각화를 위해 정규화

plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(img_A_array)
plt.title(f"1. Original Image (A)\nPredicted: {original_prediction}")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_B_array)
plt.title(f"2. Modified Image (B)\nPredicted: {modified_prediction}")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(key_K_visual_array)
plt.title("3. Adversarial Key (K)\nThe 'Secret Key'")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(restored_A_array)
plt.title(f"4. Restored Image (B - K)\nPredicted: {restored_prediction}")
plt.axis('off')

plt.tight_layout()
plt.show()