import cv2
import torch
from torchvision import transforms

from torch.utils.data import Dataset


# 定义训练和验证数据加载器
class DataLoader_Train(Dataset):
    # 定义数据增强模块
    train_val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),                              # 转tensor
            transforms.Resize((256, 256), antialias=False),     # 调整大小
            transforms.RandomCrop(224),                         # 随机裁剪
            transforms.RandomHorizontalFlip(),                  # 随机水平反转
            transforms.RandomVerticalFlip(),                    # 随机垂直反转
        ]
    )
    # 定义数据标准化模块
    train_val_transforms_Norm = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    # 数据标准化，即均值为0，标准差为1
        ]
    )

    def __init__(self, file_list_A, file_list_B, file_list_GT):
        self.file_list_A = file_list_A                      # 源图像A
        self.file_list_B = file_list_B                      # 源图像B
        self.file_list_GT = file_list_GT                    # 真实决策图
        self.transform1 = self.train_val_transforms         # 数据增强模块
        self.transform2 = self.train_val_transforms_Norm    # 数据标准化模块

    # 返回数据集长度
    def __len__(self):
        if len(self.file_list_A) == len(self.file_list_B) == len(self.file_list_GT):
            self.filelength = len(self.file_list_A)
            return self.filelength

    # 返回图像个体
    def __getitem__(self, idx):
        # 固定随机种子
        seed = torch.random.seed()
        # 图像路径
        imgA_path = self.file_list_A[idx]                               # 源图像A路径列表
        imgB_path = self.file_list_B[idx]                               # 源图像B路径列表
        imgGT_path = self.file_list_GT[idx]                             # 真实决策图GT路径列表
        # 图A
        img_A = cv2.cvtColor(cv2.imread(imgA_path), cv2.COLOR_BGR2RGB)  # 读取图A并转RGB
        torch.random.manual_seed(seed)                                  # 超控随机种子
        img_A = self.transform1(img_A)                                  # 执行数据增强
        imgA_transformed = self.transform2(img_A)                       # 执行数据标准化
        # 图B
        img_B = cv2.cvtColor(cv2.imread(imgB_path), cv2.COLOR_BGR2RGB)  # 读取图B并转RGB
        torch.random.manual_seed(seed)                                  # 超控随机种子
        img_B = self.transform1(img_B)                                  # 执行数据增强
        imgB_transformed = self.transform2(img_B)                       # 执行数据标准化
        # 真实决策图GT
        img_GT = cv2.imread(imgGT_path, cv2.IMREAD_GRAYSCALE)           # 读取灰度真实决策图GT
        torch.random.manual_seed(seed)                                  # 超控随机种子
        imgGT_transformed = self.transform1(img_GT)                     # 执行数据增强

        return imgA_transformed, imgB_transformed, imgGT_transformed