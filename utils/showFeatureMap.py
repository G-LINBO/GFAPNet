import time
import numpy as np
import matplotlib.pyplot as plt

def draw_features_1(A, savename):
    feature_map_A = A[0, :3].detach().cpu().permute(1, 2, 0).numpy()
    feature_map_A = (feature_map_A - feature_map_A.min()) / (feature_map_A.max() - feature_map_A.min())
    # 获取当前日期时间
    # 获取当前时间戳
    timestamp = time.time()
    plt.imsave(savename + '_' + str(timestamp) + '.png', feature_map_A)


def draw_features_fin(A, savename):
    feature_map_A = A[0, 0].detach().cpu()  # (1, H, W) → (H, W)
    feature_map_A = (feature_map_A - feature_map_A.min()) / (feature_map_A.max() - feature_map_A.min())
    plt.imsave(savename + '.png', feature_map_A, cmap='gray')  # Save as grayscale


def draw_features(width, height, x, savename):
    x = x.cpu().detach().numpy()
    # 创建指定大小的画布
    fig = plt.figure(figsize=(16, 16))
    # 调整子图布局，设置边距和子图间距
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        # 划分网格并选择子图，这里原代码里的 j 未定义，推测可能是笔误，按合理逻辑用 i 计算子图位置
        plt.subplot(height, width, i + 1)
        plt.axis('off')  # 关闭坐标轴显示
        img = x[0, i, :, :]  # 从 x 中获取对应的数据切片
        pmin = np.min(img)  # 获取图像数据最小值
        pmax = np.max(img)  # 获取图像数据最大值
        # 对图像数据进行归一化处理，避免除零，加小量 0.000001
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')  # 以灰度图形式显示图像
        print("{}/{}".format(i, width * height))  # 打印当前处理进度
    # 保存绘制的图像到指定路径，设置 dpi 为 100
    fig.savefig(savename, dpi=100)
    fig.clf()  # 清空当前画布
    plt.close()  # 关闭绘图窗口
