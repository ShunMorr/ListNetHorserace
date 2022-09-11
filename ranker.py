import random

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import List
import pickle
import datetime
from tqdm import tqdm

from matplotlib import pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def softmax(t: Tensor, index: int):
    denominator = torch.sum(torch.exp(t))
    return torch.exp(t[index]) / denominator


def listnet_loss(predict: Tensor, target: Tensor):
    p_list_for_query = []
    _target = target.clone()
    _predict = predict.clone()
    _p_target = 1
    _p_predict = 1
    for i in range(3):
        _target_index = torch.argmax(_target).item()
        _predict_index = torch.argmax(_predict).item()
        softmax(_target, _target_index)
        _p_target = _p_target * softmax(_target, _target_index)
        _p_predict = _p_predict * softmax(_predict, _predict_index)
        p_list_for_query.append(-_p_target * torch.log(_p_predict))
        _target = torch.cat([_target[0:_target_index], _target[_target_index + 1:]])
        _predict = torch.cat([_predict[0:_predict_index], _predict[_predict_index + 1:]])
    p = torch.cat(p_list_for_query)
    return torch.sum(p)


class ListNet(nn.Module):
    def __init__(self):
        super(ListNet, self).__init__()
        self.layer1 = nn.Linear(15, 120)
        self.layer2 = nn.Linear(120, 24)
        self.layer3 = nn.Linear(24, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = self.layer3(x)
        return x


def ndcg(y_true, y_pred):
    def dcg(y_true, y_pred):
        _, argsort = torch.sort(y_pred, descending=True, dim=0)
        ys_true_sorted = y_true[argsort]
        ret = 0
        for i, l in enumerate(ys_true_sorted, 1):
            ret += (2 ** l - 1) / np.log2(1 + i)
        return ret

    ideal_dcg = dcg(y_true, y_true)
    pred_dcg = dcg(y_true, y_pred)
    return pred_dcg / ideal_dcg


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x_filename, y_filename):
        super().__init__()
        with open(x_filename, "rb") as f:
            self.x_list = pickle.load(f)
        with open(y_filename, "rb") as f:
            self.y_list = pickle.load(f)

    def __getitem__(self, index: int):
        x = torch.Tensor(self.x_list[index])
        y = torch.Tensor(self.y_list[index])
        return x, y.shape[0] - y

    def __len__(self):
        return len(self.y_list)


def run(dataset: Dataset, index_list_train: List, index_list_valid: List):
    net = ListNet()
    epochs = 10
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor=0.01, total_iters=30000)
    losses = torch.Tensor([0])
    max_ndcg = 0
    for epoch in range(epochs):
        for i, index in enumerate(tqdm(index_list_train)):
            opt.zero_grad()
            x_train, y_train = dataset.__getitem__(index)
            x_train.to(DEVICE)
            y_train.to(DEVICE)
            predictions = net(x_train)
            query_loss = listnet_loss(predictions, y_train)
            if torch.isnan(query_loss):
                return
            # query_loss.backward()
            losses += query_loss
            if i % 10 == 9:
                losses.backward()
                opt.step()
                scheduler.step()
                losses = torch.Tensor([0])
        print(f"loss:{query_loss}")

        with torch.no_grad():
            ndcgs = []
            for index in tqdm(index_list_valid):
                x_valid, y_valid = dataset.__getitem__(index)
                x_valid.to(DEVICE)
                y_valid.to(DEVICE)
                predictions = net(x_valid)
                ndcg_score = ndcg(y_valid, predictions).item()
                ndcgs.append(ndcg_score)
            ndcg_epoch = np.average(ndcg_score)
            if max_ndcg <= ndcg_epoch:
                torch.save(net.state_dict(), f"model_{datetime.date.today()}_ndcg{ndcg_epoch}.pth")
                max_ndcg = ndcg_epoch
            print(f"epoch:{epoch + 1} ndcg:{ndcg_epoch:.4f}")


if __name__ == "__main__":
    dataset = Dataset("data_x.pkl", "data_y.pkl")
    index_list = random.sample(list(range(len(dataset))), len(dataset))
    train_index_length = int(len(dataset) * 0.8)
    train_index = index_list[:train_index_length]
    valid_index = index_list[train_index_length:]
    run(dataset, train_index, valid_index)
