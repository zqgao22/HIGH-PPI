import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
# from pygcn.layers import GraphConvolution
# from dgl.nn import GraphConv, EdgeWeightNorm
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv, GCNConv
from torch_geometric.nn.pool import SAGPooling
class GIN_Net2(torch.nn.Module):
    def __init__(self, in_len=1, in_feature=64, gin_in_feature=64, num_layers=2,
                 hidden=512, use_jk=None, pool_size=1, cnn_hidden=1, train_eps=True,
                 feature_fusion=None, class_num=7):
        super(GIN_Net2, self).__init__()
        self.use_jk = use_jk
        self.train_eps = train_eps
        self.feature_fusion = feature_fusion

        self.conv1d = nn.Conv1d(in_channels=in_feature, out_channels=cnn_hidden, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm1d(cnn_hidden)
        self.biGRU = nn.GRU(cnn_hidden, cnn_hidden, bidirectional=True, batch_first=True, num_layers=1)
        self.maxpool1d = nn.MaxPool1d(pool_size, stride=pool_size)
        self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(math.floor(in_feature), gin_in_feature)

        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(gin_in_feature, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.gin_convs.append(
                GINConv(
                    nn.Sequential(
                        # nn.Linear(hidden, hidden),
                        # nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden),
                    ), train_eps=self.train_eps
                )
            )
        if self.use_jk:
            mode = 'cat'
            self.jump = JumpingKnowledge(mode)
            self.lin1 = nn.Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, class_num)

    def reset_parameters(self):

        self.conv1d.reset_parameters()
        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        for gin_conv in self.gin_convs:
            gin_conv.reset_parameters()

        if self.use_jk:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.fc2.reset_parameters()

    def forward(self, x, edge_index, train_edge_id, p=0.5):

        x = self.gin_conv1(x, edge_index)
        xs = [x]
        for conv in self.gin_convs:
            x = conv(x, edge_index)
            xs += [x]

        if self.use_jk:
            x = self.jump(xs)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=p, training=self.training)
        x = self.lin2(x)
        # x  = torch.add(x, x_)

        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]

        if self.feature_fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.mul(x1, x2)
        x = self.fc2(x)

        return x


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(7, 32)
        # self.conv2 = GraphConv(16, 13, activation=nn.ReLU())
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 64)
        self.conv5 = GCNConv(64, 64)
  
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(64)
        self.sag1 = SAGPooling(32,0.1)
        self.sag2 = SAGPooling(64,1e-4)
        # self.linear = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(0.5)
        for param in self.parameters():
            print(type(param), param.size())

        self.fc1 = nn.Linear(16, 16)
        # self.fc2 = nn.Linear(13, 13)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.tanh(x)  # 激活函数
        # x = F.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.tanh(x)
        # x = F.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.tanh(x)

        y = self.sag1(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]
        x = self.conv4(x, edge_index)
        # x = F.dropout(x)
        x = self.bn4(x)
        x = F.tanh(x)
        x = self.conv5(x, edge_index)
        # x = F.dropout(x)
        # x = self.bn5(x)
        # x = F.tanh(x)
        x = self.sag2(x, edge_index, batch = batch)
        return x[0]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ppi_model(nn.Module):
    def __init__(self):
        super(ppi_model,self).__init__()
        self.former = GCN()

        # self.former = GIN_Net1(in_len=1, in_feature=128, gin_in_feature=13, num_layers=1, pool_size=1, cnn_hidden=1)
        self.latter = GIN_Net2()

    def forward(self, batch, p_x_all, p_edge_all, edge_index, train_edge_id, p=0.5):
        edge_index = edge_index.to(device)
        batch = batch.to(torch.int64).to(device)
        x = p_x_all.to(torch.float32).to(device)
        edge = torch.LongTensor(p_edge_all.to(torch.int64)).to(device)
        # edge_index = torch.LongTensor(p_edge_all.int()).to(device)
        embs = self.former(x, edge, batch-1)
        final = self.latter(embs, edge_index, train_edge_id, p=0.5)
        return final
