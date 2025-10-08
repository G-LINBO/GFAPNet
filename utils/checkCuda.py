import torch


# 选择设备
class GPUorCPU:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'
