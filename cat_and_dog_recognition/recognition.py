"""
作者：ZZN  
日期：2023年03月29日
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
# 线性调整学习率
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def Dataloader(args):
    split_ratio = [0.6, 0.2, 0.2]
    transform = transforms.Compose([
        transforms.RandomResizedCrop(150),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset_ori = datasets.ImageFolder('./dataset/train', transform)
    train_size = int(split_ratio[0] * len(dataset_ori))
    eval_size = int(split_ratio[1] * len(dataset_ori))
    test_size = int(split_ratio[1] * len(dataset_ori))
    train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(dataset_ori,
                                                                              [train_size, eval_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader, eval_dataloader, test_dataloader


class MyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.BatchNorm2d(256), nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(),
                                   nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(),
                                   nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten())
        self.cnn = nn.Sequential(self.conv1, self.conv2, self.conv3)
        self.linear = nn.Sequential(nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=args.dropout), nn.Linear(4096, 1),
                                    nn.Sigmoid())

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x.view(x.size(0), -1))
        return x.squeeze(-1)


def train(args):
    wandb.init(
        # set the wandb project where this run will be logged
        project="CNN",

        # track hyperparameters and run metadata
        config={
            "learning_rate": args.learning_rate,
            "architecture": "CNN",
            "epochs": args.epochs,
        }
    )
    model = MyModel(args)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = model.to(device)
    train_dataloader, eval_dataloader, test_dataloader = Dataloader(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schelduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    score_best = 0

    for epoch in range(args.epochs):
        idx = 0
        model.train()
        for data, target in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{args.epochs}', leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            target = target.float()
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            print(f'loss:{loss.item()}')
            wandb.log({"epoch:": epoch, "step:": idx, "loss:": loss.item(),
                       'learning_rate': optimizer.state_dict()['param_groups'][0]['lr']})
            idx += 1
        schelduler.step()
        correct = 0
        total = 0
        for data, target in tqdm(eval_dataloader):
            model.eval()
            with torch.no_grad():
                data, target = data.to(device), target.to(device)
                output = model(data)
                label = output.round().detach()
                correct += (label == target).sum().item()
                total += target.size(0)
        if correct / total > score_best:
            score_best = correct / total
            torch.save(model.state_dict(), './model/model.pth')
        print(f'accuracy:{correct / total}')
        wandb.log({"accuracy:": correct / total})
    print(f'best accuracy:{score_best}')


def test(args):
    model = MyModel(args)
    pretrained_dict = torch.load('./model/model.pth')
    model.load_state_dict(pretrained_dict)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = model.to(device)
    train_dataloader, eval_dataloader, test_dataloader = Dataloader(args)
    correct = 0
    total = 0
    for data, target in tqdm(test_dataloader):
        model.eval()
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            output = model(data)
            label = output.round().detach()
            correct += (label == target).sum().item()
            total += target.size(0)
    print(f'accuracy:{correct / total}')




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay (L2 penalty)')
    args = parser.parse_args()
    train(args)
    # test(args)
