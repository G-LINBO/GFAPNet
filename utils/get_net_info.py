from net import Net
import torch
from thop import profile

modelN = Net()

# 2. 准备所有输入参数（需与forward的参数类型/形状匹配）
A = torch.randn(1, 3, 520, 520)  # 假设是图像输入
B = torch.randn(1, 3, 520, 520)  # 假设是图像输入

# 3. 传入所有参数（以元组形式）
flops, params = profile(modelN, inputs=(A, B))  # 关键：传入所有required参数

print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Params: {params / 1e6:.2f} M")