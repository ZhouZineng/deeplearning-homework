"""
作者：ZZN
日期：2023年04月18日
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def prepareData(args):
    datas = np.load('tang.npz', allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = torch.from_numpy(data)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    return dataloader, ix2word, word2ix, data.shape[1]


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, args):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = args.num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers)
        self.bidirectional = 2 if args.bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(2048, vocab_size)
        )

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()

        if hidden is None:
            h_0 = input.data.new(self.bidirectional * self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(self.bidirectional * self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        embeds = self.embedding(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = output.reshape(seq_len * batch_size, -1)
        output = self.fc(output)
        return output, hidden


def train(args):
    dataloader, ix2word, word2ix, embedding_dim = prepareData(args)
    vocab_size = len(ix2word)
    hidden_dim = args.hidden_dim
    model = PoetryModel(vocab_size, embedding_dim, hidden_dim, args)  # 词向量维度，隐藏层维度
    device = torch.device("cuda" if args.cuda else "cpu")
    model = model.to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        train_loss = 0
        model.train()
        for batch_idx, data in enumerate(dataloader):
            data = data.long().transpose(1, 0).contiguous()
            data = data.to(device)
            optimizer.zero_grad()
            input, target = data[:-1, :], data[1:, :]
            output, hidden = model(input)
            loss = loss_func(output, target.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (batch_idx + 1) % 20 == 0:
                print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                    epoch, batch_idx * len(data[1]), len(dataloader),
                           100. * batch_idx / len(dataloader), loss.item()))
        lr_scheduler.step()
        start_words = '床前明月光'
        resut = generate(model, start_words, ix2word, word2ix, 75,device)
        print(resut)
        # poetry = ''
        # for word in resut:
        #     poetry += word
        #     if word == '。' or word == '!':
        #         poetry += '\n'

        # print(poetry)
    torch.save(model.state_dict(), './model/model.pth')



def generate(model, start_words, ix2word, word2ix, max_gen_len,device=torch.device("cpu")):
    # 读取唐诗的第一句
    results = list(start_words)
    start_word_len = len(start_words)

    # 设置第一个词为<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.to(device)
    hidden = None

    # 生成唐诗
    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        # 读取第一句
        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        # 生成后面的句子
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        # 结束标志
        if w == '<EOP>':
            del results[-1]
            break

    return results


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=700)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--dropout', default=0.01, type=float)
    parser.add_argument('--hidden_dim', default=512, type=int)#default=512
    parser.add_argument('--num_layers', default=4, type=int)#default=3
    parser.add_argument('--bidirectional', default=False, type=bool)

    args = parser.parse_args()
    train(args)


    dataloader, ix2word, word2ix, embedding_dim = prepareData(args)
    vocab_size = len(ix2word)
    hidden_dim = args.hidden_dim
    model =PoetryModel(vocab_size, embedding_dim, hidden_dim, args)
    pretrained_dict = torch.load('./model/model.pth')
    model.load_state_dict(pretrained_dict)
    start_words = '我喜爱深度学习'
    resut = generate(model, start_words, ix2word, word2ix, 75)
    poetry = ''
    for word in resut:
        poetry += word
        if word == '。' or word == '!':
            poetry += '\n'

    print(poetry)
    # generate()
