import torch
import numpy as np
from skimage.morphology import remove_small_objects

from utils.checkCuda import GPUorCPU

DEVICE = GPUorCPU.DEVICE


# 将输入张量二值化处理，阈值默认0.5
def Binarization(tensor, threshold=0.5):
    ones = torch.ones_like(tensor)
    zeros = torch.zeros_like(tensor)
    return torch.where(tensor > threshold, ones, zeros)


# 移除二值图像中的小区域（同时处理前景和背景中的小区域）
def RemoveSmallArea(binary_tensor, min_size=None, threshold=0.001):
    """
    参数:
        binary_tensor (torch.Tensor): 输入的二值图像张量 (B,C,H,W)
        min_size (int): 最小区域尺寸阈值，若为None则根据图像尺寸自动计算
        threshold (float): 自动计算阈值时使用的比例系数
    """
    # 确保输入是PyTorch张量
    if not isinstance(binary_tensor, torch.Tensor):
        raise TypeError("输入必须是torch.Tensor类型")
    # 保存原始设备和维度信息
    original_device = binary_tensor.device
    batch_size, channels, height, width = binary_tensor.shape
    # 计算最小区域尺寸（如果未指定）
    if min_size is None:
        min_size = threshold * height * width
    # 将张量转换为NumPy数组进行处理
    binary_array = binary_tensor.detach().cpu().numpy().astype(np.bool)
    # 移除前景中的小区域
    filtered_fg = remove_small_objects(binary_array, min_size=min_size)
    # 移除背景中的小区域（通过反转图像实现）
    inverted_bg = (1 - filtered_fg).astype(np.bool)
    filtered_bg = remove_small_objects(inverted_bg, min_size=min_size)
    final_filtered = 1 - filtered_bg
    final_filtered = final_filtered.astype(np.float32)
    # 将处理后的数组转回张量并放回原设备
    result_tensor = torch.from_numpy(final_filtered)
    result_tensor = result_tensor.to(original_device)

    return result_tensor



