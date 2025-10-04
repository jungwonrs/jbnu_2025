import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import requests
import json
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as TF
import io
# timm 라이브러리 임포트
try:
    import timm
except ImportError:
    timm = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def jpeg_compress(tensor_image, quality=75):
    pil_image = TF.to_pil_image(tensor_image.cpu().detach())
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    pil_image = Image.open(buffer)
    return TF.to_tensor(pil_image)

def gaussian_blur(tensor_image, kernel_size=3):
    return TF.gaussian_blur(tensor_image, kernel_size=kernel_size)

def gaussian_noise(tensor_image, std=0.05):
    noise = torch.randn_like(tensor_image) * std
    return tensor_image + noise

def setup_model_and_labels(model_name="resnet50"):
    print(f' {model_name} 모델을 준비중')
    
    # --- ⭐ 2. 모델 로드 로직 수정: timm 연동 ---
    try:
        # 먼저 torchvision에서 모델을 찾아봅니다.
        if model_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1')
        elif model_name == 'vgg16':
            model = models.vgg16(weights='IMAGENET1K_V1')
        elif model_name == 'vit_b_16':
            model = models.vit_b_16(weights='IMAGENET1K_V1')
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        elif model_name == 'convnext_tiny':
            model = models.convnext_tiny(weights='IMAGENET1K_V1')
        elif model_name == 'resnext50_32x4d':
            model = models.resnext50_32x4d(weights='IMAGENET1K_V1')
        elif model_name == 'wide_resnet50_2':
            model = models.wide_resnet50_2(weights='IMAGENET1K_V1')
        elif model_name == 'densenet121':
            model = models.densenet121(weights='IMAGENET1K_V1')
        elif model_name == 'inception_v3':
            model = models.inception_v3(weights='IMAGENET1K_V1')
        else:
            # torchvision에 없으면 timm에서 찾아서 불러옵니다.
            print(f" -> torchvision에 없는 모델. timm 라이브러리에서 '{model_name}'을(를) 탐색합니다...")
            model = timm.create_model(model_name, pretrained=True)

    except Exception as e:
        # timm에서도 모델을 찾지 못하면 에러를 발생시킵니다.
        raise ValueError(f"'{model_name}' 모델을 torchvision 또는 timm에서 로드할 수 없습니다. 모델 이름을 확인해주세요. 에러: {e}")
    
    model.to(DEVICE)
    model.eval()

    LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    LOCAL_LABELS_PATH = "imagenet_class_index.json"
    
    if os.path.exists(LOCAL_LABELS_PATH):
        with open(LOCAL_LABELS_PATH, "r") as f:
            imagenet_labels = json.load(f)
    else:
        print(f" -> 로컬 라벨 파일이 없어 '{LABELS_URL}'에서 다운로드합니다.")
        try:
            response = requests.get(LABELS_URL)
            response.raise_for_status()
            data = response.json()
            with open(LOCAL_LABELS_PATH, "w") as f:
                json.dump(data, f)
            print(f" -> 라벨 파일을 '{LOCAL_LABELS_PATH}'에 저장했습니다.")
            imagenet_labels = data
        except requests.exceptions.RequestException as e:
            print(f"FATAL: 라벨 파일을 다운로드할 수 없습니다. 인터넷 연결을 확인해주세요. 에러: {e}")
            exit()
    
    imagenet_labels = {k: v[1] for k, v in imagenet_labels.items()}

    print("준비완료")
    return model, imagenet_labels

def preprocess_image(pil_image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(pil_image)

def tensor_to_image_array(tensor):
    img_array = tensor.clone().detach().cpu().numpy().squeeze()
    img_array = img_array.transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = std * img_array + mean
    img_array = np.clip(img_array, 0, 1)
    return (img_array * 255).astype(np.uint8)

def evaluate_image_tensor(image_tensor, model, labels):
    input_batch = image_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    label = labels[str(top1_catid.item())]
    prob = top1_prob.item() * 100
    return label, prob

def calculate_image_metrics(img_tensor1, img_tensor2):
    img_array1 = tensor_to_image_array(img_tensor1)
    img_array2 = tensor_to_image_array(img_tensor2)
    psnr_score = psnr(img_array1, img_array2, data_range=255)
    ssim_score = ssim(img_array1, img_array2, channel_axis=2, data_range=255)
    return psnr_score, ssim_score

def save_tensor_as_image(tensor, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img_array = tensor_to_image_array(tensor)
    pil_image = Image.fromarray(img_array)
    pil_image.save(file_path)
    print(f"이미지가 '{file_path}'에 저장됨")