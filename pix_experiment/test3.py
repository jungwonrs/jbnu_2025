import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import requests
import json
import matplotlib.pyplot as plt

print ("평가 모델 준비")

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
model.to(device)
model.eval()

LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
response = requests.get(LABELS_URL)
imagenet_labels = {k: v[1] for k, v in json.loads(response.text).items()}

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def tensor_to_image_array(tensor):
    img_array = tensor.clone().detach().cpu().numpy().squeeze()
    img_array = img_array.transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = std * img_array + mean
    img_array = np.clip(img_array, 0, 1)
    return (img_array * 255).astype(np.uint8)

def evaluate_image(image_tensor, model, labels):
    input_batch = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top1_prob, top1_catid = torch.topk(probabilities, 1)
    label = labels[str(top1_catid.item())]
    prob = top1_prob.item() * 100
    return f"{label} ({prob:.2f}%)"

print("원본 이미지")
IMG_URL = "https://cdn.pixabay.com/photo/2016/12/13/05/15/puppy-1903313_1280.jpg"
img_A_pil = Image.open(requests.get(IMG_URL, stream=True).raw).convert("RGB")

#IMG_PATH = r"C:~5.jpg" 
#img_A_pil = Image.open(IMG_PATH).convert("RGB")


img_A_tensor = preprocess(img_A_pil)
print("이미지 호출 완료")

original_prediction = evaluate_image(img_A_tensor, model, imagenet_labels)
print (f"원본 (A) 평가: {original_prediction}")

# attack(?)
print ("attack setting")

epsilon = 0.03 # 강도
alpha = 0.003 # step size
num_iter = 40 # 횟수

target_label_idx = 9 #타조?
target = torch.tensor([target_label_idx]).to(device)
loss_fn = nn.CrossEntropyLoss()

perturbed_tensor = img_A_tensor.clone().detach().to(device)
img_A_tensor_device = img_A_tensor.clone().detach().to(device)

for i in range(num_iter):
    perturbed_tensor.requires_grad = True
    output = model(perturbed_tensor.unsqueeze(0))

    loss = loss_fn(output, target)
    model.zero_grad()
    loss.backward()

    attack_update = alpha * perturbed_tensor.grad.sign()

    perturbed_tensor = perturbed_tensor.detach() - attack_update

    eta = torch.clamp(perturbed_tensor - img_A_tensor, -epsilon, epsilon)

    perturbed_tensor = img_A_tensor_device + eta

print(f" -> {num_iter} 반복 완료")
adversarial_tensor = perturbed_tensor.detach().cpu()

# result
print("원본 및 공격 받은 이미지 평가")
print(f"   ->  원본(A) 평가 결과: {original_prediction}")
modified_prediction = evaluate_image(adversarial_tensor, model, imagenet_labels)
print(f"   ->  공격받은(B) 평가 결과: {modified_prediction}")


# --- 5. 시각화 ---
print("\n✅ 5. 결과 시각화...")
img_A_array = tensor_to_image_array(img_A_tensor)
adversarial_array = tensor_to_image_array(adversarial_tensor)
# '적대적 키(잔차)' 계산 및 시각화용으로 정규화
adversarial_key = adversarial_tensor - img_A_tensor
key_visual_array = tensor_to_image_array(adversarial_key / epsilon)


plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(img_A_array)
plt.title(f"1. Original Image (A)\nPredicted: {original_prediction}")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(adversarial_array)
plt.title(f"2. Adversarial Image (B)\nPredicted: {modified_prediction}")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(key_visual_array)
plt.title("3. Adversarial Key (K)")
plt.axis('off')

# 복원 로직 추가
restored_tensor = adversarial_tensor - adversarial_key
restored_array = tensor_to_image_array(restored_tensor)
restored_prediction = evaluate_image(restored_tensor, model, imagenet_labels)
plt.subplot(2, 2, 4)
plt.imshow(restored_array)
plt.title(f"4. Restored Image (B - K)\nPredicted: {restored_prediction}")
plt.axis('off')

plt.tight_layout()
plt.show()