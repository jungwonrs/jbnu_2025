import torch, platform
device = torch.device('cuda' if torch.cuda.is_available() else 'GPU')
print('Using device ➜', device)

device2 = torch.device('cuda' if torch.cuda.is_available() else 'CPU')
print('Using device ➜', device2)
