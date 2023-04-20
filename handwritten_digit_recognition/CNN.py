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


# start a new wandb run to track this script


def Dataloader(args):
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
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    return train_dataloader, test_dataloader


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 5, 1, 2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.cnn = nn.Sequential(self.conv1, self.conv2)
        self.linear = nn.Sequential(nn.Linear(7 * 7 * 32, 10), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x.view(x.size(0), -1))
        return x


def train(args):
    model = CNN()
    device = torch.device("cuda" if args.cuda else "cpu")
    model = model.to(device)
    train_dataloader, testdataloader = Dataloader(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_func = nn.CrossEntropyLoss()
    # print(model)

    model.train()

    for epoch in range(args.epochs):
        for step, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            print(x.dtype, y.dtype, output.dtype)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print("epoch:", epoch, "step:", step, "loss:", loss.item())
                # wandb.log({"epoch:": epoch, "step:": step, "loss:": loss.item()})
    torch.save(model.state_dict(), './model/model.pth')
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

    # wandb.finish()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=12, help='number of epochs to train for')
    parser.add_argument('--dataset_name', default='mnist', help='name of the dataset')
    parser.add_argument('--cuda', default=True, help='enables cuda')
    args = parser.parse_args()
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="CNN",
    #
    #     # track hyperparameters and run metadata
    #     config={
    #         "learning_rate": args.learning_rate,
    #         "architecture": "CNN",
    #         "dataset": "CIFAR-100",
    #         "epochs": args.epochs,
    #     }
    # )
    train(args)
