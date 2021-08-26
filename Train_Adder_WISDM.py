import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os
import torch.nn as nn
from scipy import stats
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.autograd import Function
import math
import torch.nn.functional as F
import argparse
from torchsummary import summary
from tensorboardX import SummaryWriter
from adder.adder_gai import Adder2D
#定义Summary_Writer
writer = SummaryWriter('./Result')

parser = argparse.ArgumentParser(description='train-WISDM-CNN')
parser.add_argument('--output_dir', type=str, default='saved_model/')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

acc = 0
acc_best = 0
train_tensorboard_step = 1

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            Adder2D(1, 128, (6, 1), (2, 1), 0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            Adder2D(128, 256, (6, 1), (2, 1), 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            Adder2D(256, 384, (6, 2), (2, 1), 0),
            nn.BatchNorm2d(384),
            nn.ReLU(True)
        )
        self.flc = nn.Linear(16128, 6)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.flc(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()
        x = F.normalize(x.cuda())
        return x


# prepare the wisdm data
train_x = torch.from_numpy(np.load('./data/wisdm/train_x.npy', encoding="latin1")).float()
train_y = torch.from_numpy(np.load('./data/wisdm/train_y.npy', encoding="latin1")).long()
test_x = torch.from_numpy(np.load('./data/wisdm/test_x.npy', encoding="latin1")).float()
test_y = torch.from_numpy(np.load('./data/wisdm/test_y.npy', encoding="latin1")).long()

# train_x = train_x.permute(0, 2, 1).contiguous()
train_x = train_x.unsqueeze(1)
train_y = torch.topk(train_y, 1)[1].squeeze(1)  # one-hot标签转换为普通label标签

# test_x = test_x.permute(0, 2, 1).contiguous()
test_x = test_x.unsqueeze(1)
test_y = torch.topk(test_y, 1)[1].squeeze(1)  # one-hot标签转换为普通label标签


data_train = TensorDataset(train_x, train_y)
data_test = TensorDataset(test_x, test_y)
data_train_loader = torch.utils.data.DataLoader(data_train, batch_size=256, shuffle=True, num_workers=2)
data_test_loader = torch.utils.data.DataLoader(data_test, batch_size=2560, shuffle=True, num_workers=0)

net = CNN().cuda()  # 选择nn
criterion = torch.nn.CrossEntropyLoss().cuda()
summary(net, (1, 200, 3))

optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-4)


def train(epoch):
    # adjust_learning_rate(optimizer, epoch)  #ALR for addernet
    global train_tensorboard_step
    net.train()
    loss_list, batch_list = [], []
    for i, (datas, labels) in enumerate(data_train_loader):
        datas, labels = Variable(datas).cuda(), Variable(labels).cuda()

        optimizer.zero_grad()

        output = net(datas)

        loss = criterion(output, labels)

        loss_list.append(loss.data.item())
        batch_list.append(i + 1)

        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))

        loss.backward()
        optimizer.step()

        writer.add_scalar('Train_loss', loss, train_tensorboard_step)
        train_tensorboard_step += 1


def test(epoch):
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (datas, labels) in enumerate(data_test_loader):
            datas, labels = Variable(datas).cuda(), Variable(labels).cuda()
            output = net(datas)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
    writer.add_scalar('val_loss', avg_loss, epoch)
    writer.add_scalar('val_acc', acc, epoch)


def train_and_test(epoch):
    train(epoch)
    test(epoch)
    for name, param in net.named_parameters():
        if 'bn' not in name:
            writer.add_histogram(name, param, epoch)

def main():
    epoch = 200
    for e in range(1, epoch):
        train_and_test(e)
    torch.save(net.state_dict(), args.output_dir + 'WISDM_CNN_model.pth')
    writer.close()


if __name__ == '__main__':
    main()