import os
import sys
import glob
import torch
import datetime
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from net import Net
from utils.checkCuda import GPUorCPU
from utils.dataLoader import DataLoader_Train
from utils.saveModel import SaveModel


class NetTrain:
    def __init__(self,
                 data_path='./Datasets/TrainAndVal/MFFdatasets',    # 训练数据集根目录
                 batchsize=16,          # 批大小
                 workers=8,             # 装载器数量（大于0时使用多进程执行数据预装载）
                 epochs=200,            # epoch总数
                 lr=0.0005,             # 初始学习率
                 gamma=0.88,            # 调度器迭代幅度
                 scheduler_step=1,      # 调度器迭代频率
                 patience=10,           # 早停阈值
                 save_path='./runs',    # 模型保存初始路径
                 prefix='train'         # 模型保存文件夹名
                 ):
        # 参数设置
        self.DEVICE = GPUorCPU().DEVICE     # 选择计算设备
        self.DATAPATH = data_path
        self.BATCHSIZE = batchsize
        self.WORKERS = workers
        self.EPOCHS = epochs
        self.LR = lr
        self.GAMMA = gamma
        self.SCHEDULER_STEP = scheduler_step
        self.PATIENCE = patience
        self.SAVEPATH = save_path
        self.PREFIX = prefix

    def __call__(self, *args, **kwargs):
        # 准备数据
        TRAIN_LOADER, VAL_LOADER, TRAIN_LEN, VAL_LEN = self.PrepareDataLoader(self.DATAPATH, self.BATCHSIZE)
        # 构建训练要素
        MODEL, OPTIMIZER, SCHEDULER, NUM_PARAMS = self.BuildModel(self.DEVICE, self.LR, self.SCHEDULER_STEP, self.GAMMA)
        print("============= Model =============")
        print(MODEL)
        print("\nThe number of model parameters: {} M\n".format(round(NUM_PARAMS / 10e5, 6)))
        print("=================================\n")
        print("============= Data =============")
        if self.DEVICE == "cuda":
            print(f"CUDA is available, using:  {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("CUDA is not available: using CPU")
        print("epochs:{}".format(self.EPOCHS))
        print("batch-size:{}".format(self.BATCHSIZE))
        print("num-worker:{}".format(self.WORKERS))
        print("Train Data A/B/GT: {}".format(TRAIN_LEN))
        print("Val Data A/B/GT: {}".format(VAL_LEN))
        print("Train Data Size:{} , Train Loader Amount: {}/{} = {}".format(
            TRAIN_LEN, TRAIN_LEN, self.BATCHSIZE, len(TRAIN_LOADER)))
        print("Val Data Size:{} , Val Loader Amount: {}/{} = {}".format(
            VAL_LEN, VAL_LEN, self.BATCHSIZE, len(VAL_LOADER)))
        print("================================\n")
        # 训练模型
        self.TrainingProcess(MODEL, OPTIMIZER, SCHEDULER, TRAIN_LOADER, VAL_LOADER, self.EPOCHS)

    # 准备数据集（训练、验证）
    def PrepareDataLoader(self, datapath, batchsize):
        # 读取数据集文件
        train_list_A = sorted(glob.glob(os.path.join(datapath, 'train/sourceA', '*.*')))
        train_list_B = sorted(glob.glob(os.path.join(datapath, 'train/sourceB', '*.*')))
        train_list_GT = sorted(glob.glob(os.path.join(datapath, 'train/decisionmap', '*.*')))
        val_list_A = sorted(glob.glob(os.path.join(datapath, 'val/sourceA', '*.*')))
        val_list_B = sorted(glob.glob(os.path.join(datapath, 'val/sourceB', '*.*')))
        val_list_GT = sorted(glob.glob(os.path.join(datapath, 'val/decisionmap', '*.*')))
        # 配置Dataloader
        train_data = DataLoader_Train(train_list_A, train_list_B, train_list_GT)
        val_data = DataLoader_Train(val_list_A, val_list_B, val_list_GT)
        # 实例化Dataloader
        train_loader = DataLoader(dataset=train_data,           # 已配置的Dataloader
                                  batch_size=batchsize,         # 批大小
                                  shuffle=True,                 # 是否随机洗牌（训练时务必开启）
                                  num_workers=self.WORKERS,     # 装载器数量（大于0时使用多进程执行数据预装载，windows慎用）
                                  pin_memory=False)             # 是否以“锁页”方式存入内存
        val_loader = DataLoader(dataset=val_data,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=self.WORKERS,
                                  pin_memory=False)

        return train_loader, val_loader, len(train_data), len(val_data)

    # 构建模型
    def BuildModel(self, device, lr, scheduler_step, gamma):
        model = Net().to(device)  # 实例化模型
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)   # 实例化优化器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step, gamma=gamma)    # 实例化学习调度器
        # 显示模型参数量
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

        return model, optimizer, scheduler, num_params

    # 模型训练过程
    def TrainingProcess(self, model, optimizer, scheduler, train_loader, val_loader, epochs):
        torch.backends.cudnn.benchmark = True   # cudnn库自动寻找优化路径
        scaler = torch.cuda.amp.GradScaler()    # 实例化梯度缩放，帮助混合精度计算
        save_model = SaveModel(self.SAVEPATH, self.PREFIX, self.PATIENCE)    # 实例化保存模型函数

        tqdm.write('Training start...\n')

        for epoch in range(epochs):
            # ============================ 开始训练 ============================
            epoch_loss = 0      # 记录每个训练epoch的总损失
            epoch_accuracy = 0  # 记录每个训练epoch的总精确度
            train_loader_tqdm = tqdm(train_loader, colour='red', leave=False, file=sys.stdout)    # 实例化一个tqdm对象
            for A, B, GT in train_loader_tqdm:  # 调用tqdm对象，实现训练进度可视化
                A = A.to(self.DEVICE)
                B = B.to(self.DEVICE)
                GT = GT.to(self.DEVICE)
                optimizer.zero_grad()   # 优化器梯度置0
                # 自动混合精确训练
                with torch.autocast(device_type=self.DEVICE, dtype=torch.float16):  # 混合精度训练模型并计算loss
                    NetOut = model(A, B)            # 数据喂到模型，模型返回预测值
                    # 求损失
                    l1_loss = nn.L1Loss()
                    loss = l1_loss(GT, NetOut)
                scaler.scale(loss).backward()   # loss回传
                scaler.step(optimizer)          # 梯度反向传导
                scaler.update()                 # 梯度缩放更新
                # 损失和准确率计算和进度可视化
                epoch_loss += loss / len(train_loader)  # 一个epoch中的一次训练的loss
                epoch_accuracy += (1 - loss) / len(train_loader)    # 一个epoch中的一次训练的acc
                # 配置进度条，显示实时loss等数据
                train_loader_tqdm.set_description("[%s] Epoch %s" % (str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')), str(epoch + 1)))
                train_loader_tqdm.set_postfix(loss=float(loss), acc=1 - float(loss))
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            # ============================ 训练结束 ============================

            # ============================ 开始验证 ============================
            with torch.no_grad():       # 关闭梯度计算
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                val_loader_tqdm = tqdm(val_loader, colour='green', leave=False, file=sys.stdout)
                for A, B, GT in val_loader_tqdm:
                    A = A.to(self.DEVICE)
                    B = B.to(self.DEVICE)
                    GT = GT.to(self.DEVICE)
                    NetOut = model(A, B)
                    # 求损失
                    l1_loss = nn.L1Loss()
                    loss = l1_loss(GT, NetOut)
                    # 损失计算和进度可视化
                    epoch_val_accuracy += (1 - loss) / len(val_loader)
                    epoch_val_loss += loss / len(val_loader)
                    val_loader_tqdm.set_description("[Validating...] Epoch %s" % str(epoch + 1))
                    val_loader_tqdm.set_postfix(loss=float(loss), acc=1 - float(loss))
            # ============================ 验证结束 ============================

            # 输出损失和准确率信息
            tqdm.write(f"[{str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}] Epoch {epoch + 1} - "
                       f"(loss :{epoch_loss:.4f} - acc:{epoch_accuracy:.4f}) - (val_loss :{epoch_val_loss:.4f} - "
                       f"val_acc:{epoch_val_accuracy:.4f})")

            # 动态更新学习率
            scheduler.step()  # 调度器更新

            # 保存模型并判断是否需要提前停止训练
            save_model(epoch_val_loss, epoch_val_accuracy, model)
            if save_model.early_stop:
                print("\nTraining stopped early...")
                print(f"Actual training epochs:{epoch + 1}/{self.EPOCHS}")
                break  # 跳出迭代，结束训练

        # 训练结束
        if not save_model.early_stop:
            print('\nTraining stop...')
            print(f"Actual training epochs:{self.EPOCHS}")
        save_model._print_model_data()  # 输出模型数据
        torch.cuda.empty_cache()


if __name__ == '__main__':
    train = NetTrain()
    train()
