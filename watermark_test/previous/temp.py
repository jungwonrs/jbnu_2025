import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("arch list:", torch.cuda.get_arch_list())   # sm_120 들어 있어야 OK
print("device   :", torch.cuda.get_device_name(0))
