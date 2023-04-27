"""
作者：ZZN  
日期：2023年03月28日
"""
import argparse
import os

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import wandb

#数据集加载及训练集和测试集划分
def Dataloader(args):
    # 加载 MNIST 训练集和测试集
    train_data = torchvision.datasets.MNIST(
        root=f'./{args.dataset_name}',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_data = torchvision.datasets.MNIST(
        root=f'./{args.dataset_name}',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    # 创建训练和测试的 dataloader
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    return train_dataloader, test_dataloader

#定义模型
class CNN(nn.Module):
    def __init__(self,args):
        super().__init__()
        # 定义模型模块（layer），包括两个卷积层和一个全连接层，并通过 ReLU 激活函数进行非线性变换
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 5, 1, 2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU())
        # 将两个卷积层连成一个整体
        self.cnn = nn.Sequential(self.conv1, self.conv2,self.conv3)
        # 创建全连接层，将原始二维图像数据转为一维向量数据，最后通过 softmax 得到结果
        self.linear = nn.Sequential(nn.Linear(7 * 7 * 64, 7 * 7 * 32),nn.ReLU(),nn.Dropout(args.dropout),nn.Linear(7 * 7 * 32, 10), nn.Softmax(dim=1))

    def forward(self, x):
        # CNN 传播过程
        x = self.cnn(x)
        # 将二维数据展开为一位向量
        x = self.linear(x.view(x.size(0), -1))
        return x

#模型 训练
def train(args):
    model = CNN(args)
    # 指定训练使用的设备是 CPU 还是 GPU
    device = torch.device("cuda" if args.cuda else "cpu")
    # 迁移模型到指定设备
    model = model.to(device)
    train_dataloader, testdataloader = Dataloader(args)
    # 设定 Adam 优化器和交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # 更新学习率
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_func = nn.CrossEntropyLoss()
    # print(model)
    model.train()
    for epoch in range(args.epochs):
        for step, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            # 前向传播 + 计算 loss
            output = model(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            # 利用梯度进行反向传播
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print("epoch:", epoch, "step:", step, "loss:", loss.item(),"lr:", optimizer.param_groups[0]['lr'])
                # wandb.log({"epoch:": epoch, "step:": step, "loss:": loss.item(),'lr:': optimizer.param_groups[0]['lr']})
        lr_scheduler.step()
    torch.save(model.state_dict(), './model.pth')
    correct = 0
    total = 0
    for x, y in testdataloader:
        x = x.to(device)
        output = model(x)
        pred_y = torch.max(output.detach().cpu(), 1)[1].data.numpy()
        total += y.size(0)
        correct += (pred_y == y.data.numpy()).sum()
    print("accuracy:", correct / total)
    # wandb.log({"accuracy:": correct / total})
    #
    # wandb.finish()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--dataset_name', default='mnist', help='name of the dataset')
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('--dropout', default=0.1, help='dropout rate')
    args = parser.parse_args()

    #用来打印日志，如果在其它机器跑可以注释掉
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="CNN",
    #
    #     # track hyperparameters and run metadata
    #     config={
    #         "learning_rate": args.learning_rate,
    #         "architecture": "CNN",
    #         "dataset": "Minst",
    #         "epochs": args.epochs,
    #     }
    # )
    train(args)
