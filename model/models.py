import math

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, LeakyReLU
from torch_geometric.nn import GINConv, global_add_pool, GCNConv


def trans_adj_to_edge_index(adj):
    adj = scipy.sparse.coo_matrix(adj.cpu().detach().numpy())
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    adj = torch.LongTensor(indices)
    return adj

class VGAE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, device):
        super(VGAE, self).__init__()
        self.device = device
        self.input_dim = nfeat
        self.hidden_dim = nhid
        self.num_gc_layers = nlayer
        self.base_gcn = Encoder(self.input_dim, self.hidden_dim, self.num_gc_layers, device)
        self.gcn_mean = Encoder(self.hidden_dim * self.num_gc_layers, self.hidden_dim, self.num_gc_layers, device)
        self.gcn_logstddev = Encoder(self.hidden_dim * self.num_gc_layers, self.hidden_dim, self.num_gc_layers, device)
        self.base_global_d = FF(self.hidden_dim * self.num_gc_layers)
        self.mean_global_d = FF(self.hidden_dim * self.num_gc_layers)
        self.logstddev_global_d = FF(self.hidden_dim * self.num_gc_layers)

    def encode(self, x, edge_index, batch):
        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones(batch.shape[0])
        _, hidden = self.base_gcn(x, edge_index, batch)
        hidden = self.base_global_d(hidden)
        _, self.mean = self.gcn_mean(hidden, edge_index, batch)
        self.mean = self.mean_global_d(self.mean)
        _, self.logstd = self.gcn_logstddev(hidden, edge_index, batch)
        self.logstd = self.logstddev_global_d(self.logstd)
        gaussian_noise = torch.randn(batch.shape[0], self.hidden_dim * self.num_gc_layers).to(self.device)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, x, edge_index, batch):
        z = self.encode(x, edge_index, batch)
        A_pred = dot_product_decode(z)
        return A_pred

    def latent_loss(self, z_mean, z_stddev):
        kl_divergence = 0.5 * torch.sum(torch.exp(z_stddev) + torch.pow(z_mean, 2) - 1. - z_stddev)
        return kl_divergence / z_mean.size(0)

    def loss(self, A_pred, A, logstd, mean):

        e_loss1 = F.binary_cross_entropy(A_pred, A)
        kl_loss_edge = self.latent_loss(mean, logstd)
        loss = e_loss1 + 0.001 * kl_loss_edge
        return loss


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)



class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class FF_baseline(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias = False),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim, bias = False),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim, bias = False),
            nn.LeakyReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim, bias = False)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class Encoder_baseline(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, device):
        super(Encoder_baseline, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.device = device

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim, bias = False), LeakyReLU(), Linear(dim, dim, bias = False))
            else:
                nn = Sequential(Linear(num_features, dim, bias = False), LeakyReLU(), Linear(dim, dim, bias = False))
            # nn = Sequential(Linear(dim, dim, bias=False))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim, eps=1e-04)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):

        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        xs = []
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(self.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(self.device)
                x, _ = self.forward(x, edge_index, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, device):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.device = device

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), LeakyReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim, eps=1e-04, affine=True, track_running_stats=True)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):

        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        xs = []
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index)

            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(self.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None or len(x.shape) == 1 or x.shape[1] == 0:
                    x = torch.ones((batch.shape[0], 1)).to(self.device)
                x, _ = self.forward(x, edge_index, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class GraphConv(nn.Module):
    def __init__(self):
        super(GraphConv, self).__init__()

    def forward(self, x, adj):
        x = torch.mm(adj, x)
        return x


class shared_GIN(torch.nn.Module):
    def __init__(self, num_features, latent_dim, num_gc_layers, device):
        super(shared_GIN, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.device = device
        self.bns = torch.nn.ModuleList()
        self.nns = torch.nn.ModuleList()
        self.embedding_dim = latent_dim * num_gc_layers
        self.gin_conv = GraphConv()
        for i in range(self.num_gc_layers):
            bn = torch.nn.BatchNorm1d(latent_dim, eps=1e-04, affine=False, track_running_stats=True)
            if i:
                nn = Sequential(Linear(latent_dim, latent_dim), LeakyReLU(), Linear(latent_dim, latent_dim))
            else:
                nn = Sequential(Linear(num_features, latent_dim), LeakyReLU(), Linear(latent_dim, latent_dim))
            self.nns.append(nn)
            self.bns.append(bn)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)


    def forward(self, x, adj, batch):
        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        xs = []
        for i in range(self.num_gc_layers):
            x = self.gin_conv(x, adj)
            x = self.nns[i](x)
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x_global = torch.cat(xpool, 1)
        x_node = torch.cat(xs, 1)

        g_enc = x_global
        l_enc = x_node
        return g_enc, l_enc


class teacher_head(torch.nn.Module):
    def __init__(self, dim, num_gc_layers, nclass, device):
        super(teacher_head, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(dim * num_gc_layers, dim * 3)
        self.fc2 = nn.Linear(dim * 3, dim * 2)
        self.fc3 = nn.Linear(dim * 2, dim)
        self.fc4 = nn.Linear(dim, nclass)

    def forward(self, g_enc):
        x = F.leaky_relu(self.fc1(g_enc))
        x = F.leaky_relu(self.fc2(x))
        middle = F.leaky_relu(self.fc3(x))
        output = self.fc4(middle)
        return output


class student_head(torch.nn.Module):
    def __init__(self, dim, num_gc_layers, nclass, device):
        super(student_head, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(dim * num_gc_layers, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.fc3 = nn.Linear(dim, nclass)

    def forward(self, g_enc):
        x = F.leaky_relu(self.fc1(g_enc))
        middle = F.leaky_relu(self.fc2(x))
        output = self.fc3(middle)
        return output

class Graph_Representation_Learning(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, dataset_num_features, device):
        super(Graph_Representation_Learning, self).__init__()
        self.pre = torch.nn.Sequential(torch.nn.Linear(dataset_num_features, hidden_dim))
        self.device = device
        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder_baseline(hidden_dim, hidden_dim, num_gc_layers, self.device)
        self.global_d = FF_baseline(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def get_result(self, loader):
        embedding = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(self.device)
                x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
                if x is None or len(x.shape) == 1 or x.shape[1] == 0:
                    x = torch.ones((batch.shape[0], 1)).to(self.device)
                pro_x = self.forward(x, edge_index, batch)
                embedding.append(pro_x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        embedding = np.concatenate(embedding, 0)
        y = np.concatenate(y, 0)
        return embedding, y

    def forward(self, x, edge_index, batch):
        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        x = self.pre(x)
        y, M = self.encoder(x, edge_index, batch)
        pro_x = self.global_d(y)
        return pro_x


class server_graph_representation_learning(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, dataset_num_features, device):
            super(server_graph_representation_learning, self).__init__()
            self.device = device

            self.embedding_dim = mi_units = hidden_dim * num_gc_layers
            self.encoder = Encoder_baseline(hidden_dim, hidden_dim, num_gc_layers, self.device)
            self.global_d = FF_baseline(self.embedding_dim)




class Encoder_fedstar(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, device):
        super(Encoder_fedstar, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.device = device

        self.Whp = torch.nn.Linear(dim + dim, dim)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs_s = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim + dim, dim, bias=False), LeakyReLU(), Linear(dim, dim, bias=False))
            else:
                nn = Sequential(Linear(dim + dim, dim, bias=False), LeakyReLU(), Linear(dim, dim, bias=False))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim, eps=1e-04, affine=True, track_running_stats=True)

            self.convs_s.append(GCNConv(dim, dim))
            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch, s):
        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        xs = []
        for i in range(self.num_gc_layers):
            x = torch.cat((x, s), -1)
            x = self.convs[i](x, edge_index)
            x_local = self.bns[i](x)
            s = self.convs_s[i](s, edge_index)
            s = torch.tanh(s)
        x = self.Whp(torch.cat((x_local, s), -1))
        x = global_add_pool(x, batch)
        return x, x_local

    def get_embeddings(self, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(self.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None or len(x.shape) == 1 or x.shape[1] == 0:
                    x = torch.ones((batch.shape[0], 1)).to(self.device)
                x, _ = self.forward(x, edge_index, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

class Graph_Representation_Learning_Fedstar(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, dataset_num_features, n_se, device):
        super(Graph_Representation_Learning_Fedstar, self).__init__()
        self.pre = torch.nn.Sequential(torch.nn.Linear(dataset_num_features, hidden_dim))
        self.embedding_s = torch.nn.Linear(n_se, hidden_dim)
        self.device = device
        self.hidden_dim=hidden_dim
        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder_fedstar(hidden_dim, hidden_dim, num_gc_layers, self.device)
        self.global_d = FF_baseline(self.hidden_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def get_result(self, loader):
        embedding = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(self.device)
                x, edge_index, batch, s = data.x, data.edge_index, data.batch, data.stc_enc
                if x is None or len(x[1]) == 0:
                    x = torch.ones(batch.shape[0], 1).to(self.device)
                pro_x = self.forward(x, edge_index, batch, s)
                embedding.append(pro_x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        embedding = np.concatenate(embedding, 0)
        y = np.concatenate(y, 0)
        return embedding, y

    def forward(self, x, edge_index, batch, s):
        if x is None or len(x[1]) == 0:
            x = torch.ones(batch.shape[0], 1).to(self.device)
        x = self.pre(x)
        s = self.embedding_s(s)
        y, M = self.encoder(x, edge_index, batch, s)
        pro_x = self.global_d(y)
        return pro_x


class server_graph_representation_learning_fedstar(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, dataset_num_features,n_se, device):
            super(server_graph_representation_learning_fedstar, self).__init__()
            self.hidden_dim=hidden_dim
            self.device = device

            self.embedding_dim = mi_units = hidden_dim * num_gc_layers
            self.encoder = Encoder_fedstar(hidden_dim, hidden_dim, num_gc_layers, self.device)
            self.global_d = FF_baseline(self.hidden_dim)




