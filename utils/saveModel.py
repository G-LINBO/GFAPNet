import os
import numpy as np
import torch


# 对每次训练的模型的验证损失进行判断，保存损失值最小的模型，并且在验证损失不再改善达到一定轮次后提前终止训练，防止模型过拟合
class SaveModel:
    def __init__(self, save_path, prefix, patience=20, verbose=False, delta=0.0):
        """
        参数:
            save_path: 模型保存初始路径
            prefix: 模型保存文件夹名
            patience: 验证损失停滞的最大轮数
            verbose: 是否输出模型保存信息
            delta: 视为损失改善的最小变化量
        """
        # 配置参数
        self.save_path = save_path
        self.prefix = prefix
        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        # 状态变量
        self.early_stop = False  # 提前终止标志
        self.counter = 0  # 损失未改善的累计轮数
        self.best_score = None  # 记录最佳验证得分
        self.val_loss_min = np.Inf  # 记录最小验证损失值
        self.val_accuracy_max = 0.  # 记录最小验证损失值

        # 创建保存模型文件夹
        self.save_model_path = self._set_save_path()

    def __call__(self, val_loss, val_accuracy, model):
        """
        参数:
            val_loss: 当前验证损失值
            model: 当前训练的模型
        """
        # 转换为负损失以便比较（越大越好）
        current_score = -val_loss

        # 首次调用或发现更优模型
        if self.best_score is None:
            self.best_score = current_score
            self._save_best_model(val_loss, val_accuracy, model)
        # 损失未改善
        elif current_score < self.best_score + self.delta:
            self.counter += 1
            print(f"Loss did not increase: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        # 发现更优模型
        else:
            self.best_score = current_score
            self._save_best_model(val_loss, val_accuracy, model)
            self.counter = 0  # 重置计数器

    # 设置模型保存路径
    def _set_save_path(self):
        # 确保基础目录存在
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # 检查是否已存在无前缀的文件夹
        initial_folder = os.path.join(self.save_path, self.prefix)
        if not os.path.exists(initial_folder):
            os.makedirs(initial_folder)
            return initial_folder
        # 查找最大编号
        max_num = 0
        for entry in os.listdir(self.save_path):
            entry_path = os.path.join(self.save_path, entry)
            if os.path.isdir(entry_path) and entry.startswith(self.prefix):
                # 提取编号部分
                try:
                    num = int(entry[len(self.prefix):])
                    if num > max_num:
                        max_num = num
                except ValueError:
                    continue  # 非数字后缀的文件夹忽略
        # 创建下一个编号的文件夹
        next_folder = os.path.join(self.save_path, f"{self.prefix}{max_num + 1}")
        os.makedirs(next_folder)

        return next_folder

    # 保存模型
    def _save_best_model(self, val_loss, val_accuracy, model):
        # 记录损失值为最小损失
        self.val_loss_min = val_loss
        self.val_accuracy_max = val_accuracy
        # 输出模型保存信息
        if self.verbose:
            old_loss = self.val_loss_min
            print(f'验证损失下降 ({old_loss:.6f} → {val_loss:.6f})，保存模型...')
        # 获取保存路径：
        model_path = os.path.join(self.save_model_path, 'best.pth')
        # 保存模型参数
        torch.save(model.state_dict(), model_path)

    # 输出最终保存模型数据
    def _print_model_data(self):
        print("============================================")
        print(f"Final val loss: {self.val_loss_min:.4f}")
        print(f"Final val accuracy: {self.val_accuracy_max:.4f}")
        print("The model is saved in '{}'".format(self.save_model_path))
        print("============================================")



