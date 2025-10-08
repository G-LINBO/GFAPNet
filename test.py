import os
import cv2
import time
import torch
from torch import einsum
from torchvision import transforms

from net import Net
from utils import consistency
from utils.checkCuda import GPUorCPU


# 图像融合
def testFusion(model, device, pathA, pathB):
    # 读取图像转换格式
    inputA = cv2.imread(pathA)
    inputB = cv2.imread(pathB)
    input1 = cv2.cvtColor(inputA, cv2.COLOR_BGR2RGB)
    input2 = cv2.cvtColor(inputB, cv2.COLOR_BGR2RGB)
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
    final_fusion = inputA * D + inputB * (1 - D)  # 图像融合

    return D * 255, final_fusion


# 读取图像并保存融合图像
def process_folder_pair(model, device, folderA, folderB, output_dir):
    """处理两个文件夹中的所有图像对（基于前缀匹配）"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 获取文件夹A中的所有图像文件，并提取前缀
    image_prefixes = {}  # 存储: 前缀 -> 完整文件名
    for filename in os.listdir(folderA):
        if not os.path.isfile(os.path.join(folderA, filename)):
            continue
        # 提取前缀（去除最后一个字母和扩展名）
        # 例如: "lytro-01-A.jpg" -> "lytro-01-"
        base_name = os.path.splitext(filename)[0]
        prefix = base_name[:-1]  # 去掉最后一个字母
        image_prefixes[prefix] = filename

    # 处理每对图像
    processed_count = 0
    begin_time = time.time()
    for prefix, filenameA in image_prefixes.items():
        # 构建图像B的文件名
        # 例如: filenameA为lytro-01-A.jpg"，则filenameB则为lytro-01-B.jpg
        filenameB = prefix + 'B' + os.path.splitext(filenameA)[1]
        pathA = os.path.join(folderA, filenameA)
        pathB = os.path.join(folderB, filenameB)
        # 检查图像B是否存在
        if not os.path.exists(pathB):
            print(f"警告: {filenameB} 在文件夹B中不存在，跳过")
            continue
        # 处理图像对
        print(f"处理: {filenameA} <-> {filenameB}")
        dm_img, fusion_img = testFusion(model, device, pathA, pathB)
        # 生成输出文件名（使用前缀作为标识）
        output_prefix = prefix.rstrip('-')  # 移除可能的末尾连字符
        # 创建决策图目录和结果图目录
        os.makedirs(output_dir + "/dm_img", exist_ok=True)
        os.makedirs(output_dir + "/fusion_img", exist_ok=True)
        dm_output_path = os.path.join(output_dir + "/dm_img", f"{output_prefix}-dm.png")
        fusion_output_path = os.path.join(output_dir + "/fusion_img", f"{output_prefix}.png")

        # 保存结果
        cv2.imwrite(dm_output_path, dm_img)
        cv2.imwrite(fusion_output_path, fusion_img)
        processed_count += 1
    proc_time = time.time() - begin_time
    print('Total processing time: {:.3}s'.format(proc_time))
    print(f"处理完成: 共找到 {len(image_prefixes)} 对，成功处理 {processed_count} 对")
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
    LytroPathA = 'Datasets/Test/Lytro/sourceA'                                                  # 图像A路径
    LytroPathB = 'Datasets/Test/Lytro/sourceB'                                                  # 图像A路径
    LytroPathOutput = 'result/Lytro'                                                          # 保存结果文件夹
    process_folder_pair(model, DEVICE, LytroPathA, LytroPathB, LytroPathOutput)                 # 读取图像并保存融合结果
    # MFFW数据集
    MFFWPathA = 'Datasets/Test/MFFW/sourceA'                                                    # 图像A路径
    MFFWPathB = 'Datasets/Test/MFFW/sourceB'                                                    # 图像B路径
    MFFWPathOutput = 'result/MFFW'                                                            # 保存结果文件夹
    #process_folder_pair(model, DEVICE, MFFWPathA, MFFWPathB, MFFWPathOutput)                    # 读取图像并保存融合结果
    # Grayscale数据集
    GrayscalePathA = 'Datasets/Test/Grayscale/sourceA'                                          # 图像A路径
    GrayscalePathB = 'Datasets/Test/Grayscale/sourceB'                                          # 图像B路径
    GrayscalePathOutput = 'result/Grayscale'                                                  # 保存结果文件夹
    #process_folder_pair(model, DEVICE, GrayscalePathA, GrayscalePathB, GrayscalePathOutput)     # 读取图像并保存融合结果
    # MFI数据集
    MFIPathA = 'Datasets/Test/MFI/sourceA'        # 图像A路径
    MFIPathB = 'Datasets/Test/MFI/sourceB'  # 图像B路径
    MFIPathOutput = 'result/MFI'  # 保存结果文件夹
    #process_folder_pair(model, DEVICE, MFIPathA, MFIPathB, MFIPathOutput)  # 读取图像并保存融合结果





