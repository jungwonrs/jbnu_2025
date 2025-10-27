import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0),
      torch.cuda.get_device_capability(0))
torch.rand(1, device='cuda')   # 오류 없으면 성공