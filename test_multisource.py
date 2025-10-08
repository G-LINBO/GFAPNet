import os
import cv2
import torch
from torch import einsum
from torchvision import transforms

from net import Net
from utils import consistency
from utils.checkCuda import GPUorCPU


# 图像融合
def testFusion(model, device, A_IMG, B_IMG):
    # 读取图像转换格式
    input1 = cv2.cvtColor(A_IMG, cv2.COLOR_BGR2RGB)
    input2 = cv2.cvtColor(B_IMG, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    A = transform(input1).unsqueeze(0).to(device)
    B = transform(input2).unsqueeze(0).to(device)

    # 输出初始决策图
    NetOut = model(A, B)
    # 小区域移除得到最终决策图
    Verified_img_tensor = consistency.Binarization(NetOut, 0.5)  # 二值化
    # D = Verified_img_tensor[0]
    D = consistency.RemoveSmallArea(binary_tensor=Verified_img_tensor, threshold=0.005)[0]

    # 通过决策图进行加权图像融合
    D = einsum('c w h -> w h c', D).clone().detach().cpu().numpy()  # 转为opencv支持的whc模式
    final_fusion = A_IMG * D + B_IMG * (1 - D)  # 图像融合

    return D * 255, final_fusion


# 读取图像并保存融合图像
def process_folder_pair(model, device, folderA, folderB, folderC, output_dir):
    # 创建决策图目录和结果图目录
    # 生成输出文件名（使用前缀作为标识）
    base_name = os.path.splitext(os.path.basename(folderA))[0]
    output_prefix = base_name[:-2]  # 去掉最后一个字母和-
    os.makedirs(output_dir + "/dm_img", exist_ok=True)
    os.makedirs(output_dir + "/fusion_img", exist_ok=True)
    dm_output_path1 = os.path.join(output_dir + "/dm_img", f"{output_prefix}-dm-1.png")
    fusion_output_path1 = os.path.join(output_dir + "/fusion_img", f"{output_prefix}-1.png")
    dm_output_path = os.path.join(output_dir + "/dm_img", f"{output_prefix}-dm-fin.png")
    fusion_output_path = os.path.join(output_dir + "/fusion_img", f"{output_prefix}-fin.png")
    A = cv2.imread(folderA)
    B = cv2.imread(folderB)
    C = cv2.imread(folderC)
    dm_img1, fusion_img1 = testFusion(model, device, A, B)
    cv2.imwrite(dm_output_path1, dm_img1)
    cv2.imwrite(fusion_output_path1, fusion_img1)
    AB = cv2.imread(fusion_output_path1)
    dm_img, fusion_img = testFusion(model, device, AB, C)
    cv2.imwrite(dm_output_path, dm_img)
    cv2.imwrite(fusion_output_path, fusion_img)
    print(f"结果保存在: {output_dir}")


if __name__ == '__main__':
    # ======================== 选择设备 ========================
    DEVICE = GPUorCPU().DEVICE
    if DEVICE == "cuda":
        print(f"CUDA is available, using:  {torch.cuda.get_device_name(torch.cuda.current_device())}\n")
    else:
        print("CUDA is not available: using CPU\n")

    # ======================== 加载模型  ========================
    pathModel = 'best.pth'       # 模型路径
    model = Net().to(DEVICE)  # 加载网络结构
    model.load_state_dict(torch.load(pathModel, map_location=torch.device(DEVICE)))  # 加载模型
    model.eval()  # 验证模式

    #  ======================== 图像融合  ========================
    # Lytro数据集
    LytroPathA = 'Datasets/Test/Lytro/Triple Series/sourceA/lytro-01-A.jpg'                     # 图像A路径
    LytroPathB = 'Datasets/Test/Lytro/Triple Series/sourceB/lytro-01-B.jpg'                     # 图像B路径
    LytroPathC = 'Datasets/Test/Lytro/Triple Series/sourceC/lytro-01-C.jpg'                     # 图像C路径
    LytroPathOutput = 'result/LytroTripleSeries'                                                # 保存结果文件夹
    process_folder_pair(model, DEVICE, LytroPathA, LytroPathB, LytroPathC, LytroPathOutput)     # 读取图像并保存融合结果
