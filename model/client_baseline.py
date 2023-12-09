import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from Utils.compute_auprc import compute_auprc_baseline

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from Utils.initialize import *
from Utils.losses import *
class Client_GC_Baseline():
    def __init__(self, model, client_id, client_name, train_size, dataLoader, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}
        self.c = None
        self.R = 0

        self.gconvNames = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

    def download_from_server(self, server):
        self.gconvNames = server.W.keys()
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()
        self.R = server.R

    def download_from_server_fedstar(self, server):
        self.gconvNames = server.W.keys()
        for k in server.W:
            if '_s' in k:
                self.W[k].data = server.W[k].data.clone()
        self.R = server.R

    def cache_weights(self):
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()

    def reset(self):
        copy(target=self.W, source=self.W_old, keys=self.gconvNames)


    def local_train(self, local_epoch, normal_class, c, nu):
        """ For self-train & FedAvg """
        self.c = c
        train_stats, R = train_ad(self.model, self.dataLoader, self.optimizer, local_epoch, c, self.R, nu, normal_class,
                                  self.args.device)

        self.train_stats = train_stats
        self.R = R
        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}

        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def local_train_fedstar(self, local_epoch, normal_class, c, nu):
        """ For FedStar """
        self.c =c
        train_stats, R = train_ad_fedstar(self.model, self.dataLoader, self.optimizer, local_epoch, c, self.R, nu, normal_class, self.args.device)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()
        self.R = R

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        # grads = {key: value.grad for key, value in self.W.items()}
        # # print(self.W)
        # self.gradsNorm = torch.norm(flatten(grads)).item()
        #
        # grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        # self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def compute_weight_update(self, local_epoch, normal_class, c, nu):
        """ For GCFL """
        self.c = c
        copy(target=self.W_old, source=self.W, keys=self.gconvNames)

        train_stats, R = train_ad(self.model, self.dataLoader, self.optimizer, local_epoch, c, self.R, nu, normal_class,
                                  self.args.device)
        self.R = R
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

        self.train_stats = train_stats

        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        dWs_conv = {key: self.dW[key] for key in self.gconvNames}
        self.convDWsNorm = torch.norm(flatten(dWs_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def evaluate(self):

        return eval_ad(self.model, self.dataLoader['test'], self.args.normal_class, self.c, self.R, self.args.device)

    def local_train_prox(self, local_epoch, mu, normal_class, c, nu):
        """ For FedProx """
        self.c = c
        train_stats, R = train_ad_prox(self.model, self.dataLoader, self.optimizer, local_epoch, c, self.R, nu,
                                       normal_class, self.args.device,
                                       self.gconvNames, self.W, mu, self.W_old)
        self.R = R
        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def evaluate_prox(self, mu):
        return eval_ad_prox(self.model, self.dataLoader['test'], self.args.normal_class, self.c, self.R,
                            self.args.device, self.gconvNames, mu, self.W_old)


def copy(target, source, keys):
    for name in keys:
        target[name].data = source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()



def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])


def calc_gradsNorm(gconvNames, Ws):
    grads_conv = {k: Ws[k].grad for k in gconvNames}
    convGradsNorm = torch.norm(flatten(grads_conv)).item()
    return convGradsNorm


def train_ad(model, dataloaders, optimizer, local_epoch, c, R, nu, normal_class, device):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    aucs_test = []
    aucprs_test = []
    f1s_test = []
    recalls_test = []

    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0
        total_dist = []
        total_output = []
        for _, batch in enumerate(train_loader):
            data = batch.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch

            optimizer.zero_grad()
            pro_emb = model(x, edge_index, batch)
            svdd_loss, dist = calculate_svdd_loss(pro_emb, c, R, nu, device)
            label = data.y
            svdd_loss.backward()
            optimizer.step()
            total_loss += svdd_loss.item() * data.num_graphs
            ngraphs += data.num_graphs
            total_dist.append(dist)
            total_output.append(pro_emb)
        total_loss /= ngraphs
        total_dist = torch.cat(total_dist, dim=0)
        total_output = torch.cat(total_output, dim=0)
        if (epoch % 1 == 0) or (epoch > epochs - 10):
            R = torch.tensor(get_radius(total_dist, nu), device=device)
            test_auc, test_auprc = eval_ad(model, test_loader, normal_class, c, R, device)
            print("loss train :", total_loss)

        losses_train.append(total_loss)
        aucs_test.append(test_auc)
        aucprs_test.append(test_auprc)

    return {'trainingLosses': losses_train, 'Aucs': aucs_test, 'Aucpr': aucprs_test}, R


def eval_ad(model, test_loader, normal_class, c, R, device):
    model.eval()
    z, y = model.get_result(test_loader)
    z = torch.tensor(z).to(device)
    dist = torch.sum((z - c) ** 2, dim=1)
    scores = dist - R ** 2
    labels = np.array(y.tolist())
    test_labels = np.where(labels == normal_class, 1, 0)
    labels = np.where(labels == normal_class, 0, 1)
    scores = np.array(scores.cpu().data.numpy().tolist())

    test_auprc = compute_auprc_baseline(test_labels, scores)
    test_auc = roc_auc_score(labels, scores)
    return test_auc, test_auprc


def _prox_term(model, gconvNames, Wt):
    prox = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        # only add the prox term for sharing layers (gConv)
        if name in gconvNames:
            prox = prox + torch.norm(param - Wt[name]).pow(2)
    return prox


def train_ad_prox(model, dataloaders, optimizer, local_epoch, c, R, nu, normal_class, device, gconvNames, Ws, mu, Wt):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    convGradsNorm = []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    aucs_test = []
    aucprs_test = []
    f1s_test = []
    recalls_test = []
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0
        # R=0.
        acc_sum = 0
        total_dist = []
        total_output = []
        for _, batch in enumerate(train_loader):
            data = batch.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch

            optimizer.zero_grad()
            pro_emb = model(x, edge_index, batch)
            svdd_loss, dist = calculate_svdd_loss(pro_emb, c, R, nu, device)
            label = data.y
            svdd_loss.backward()
            optimizer.step()
            total_loss += svdd_loss.item() * data.num_graphs
            ngraphs += data.num_graphs
            total_dist.append(dist)
            total_output.append(pro_emb)
        total_loss /= ngraphs


        total_dist = torch.cat(total_dist, dim=0)
        total_output = torch.cat(total_output, dim=0)
        if (epoch % 1 == 0) or (epoch > epochs - 10):
            R = torch.tensor(get_radius(total_dist, nu), device=device)
            test_auc, test_auprc = eval_ad_prox(model, test_loader, normal_class, c, R, device,
                                                                      gconvNames, mu, Wt)
            print("loss train :", total_loss)

        losses_train.append(total_loss)
        aucs_test.append(test_auc)
        aucprs_test.append(test_auprc)

        convGradsNorm.append(calc_gradsNorm(gconvNames, Ws))

    return {'trainingLosses': losses_train, 'Aucs': aucs_test, 'Aucpr': aucprs_test, 'convGradsNorm': convGradsNorm}, R


def eval_ad_prox(model, test_loader, normal_class, c, R, device, gconvNames, mu, Wt):
    model.eval()
    z, y = model.get_result(test_loader)
    z = torch.tensor(z).to(device)
    label_score = []
    dist = torch.sum((z - c) ** 2, dim=1)
    scores = dist - R ** 2
    labels = np.array(y.tolist())
    labels = np.where(labels == normal_class, 0, 1)
    test_labels = np.where(labels == normal_class, 1, 0)
    scores = np.array(scores.cpu().data.numpy().tolist())
    test_auprc = compute_auprc_baseline(test_labels, scores)
    test_auc = roc_auc_score(labels, scores)
    return test_auc, test_auprc


def train_ad_fedstar(model, dataloaders, optimizer, local_epoch, c, R, nu, normal_class, device):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    aucs_test = []
    aucprs_test = []
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0
        total_dist = []
        total_output = []
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            data = batch.to(device)
            x, edge_index, batch, s = data.x, data.edge_index, data.batch, data.stc_enc
            pro_emb = model(x, edge_index, batch, s)
            svdd_loss, dist = calculate_svdd_loss(pro_emb, c, R, nu, device)
            label = data.y
            svdd_loss.backward()
            optimizer.step()
            total_loss += svdd_loss.item() * data.num_graphs
            ngraphs += data.num_graphs
            total_dist.append(dist)
            total_output.append(pro_emb)
        total_loss /= ngraphs
        total_dist = torch.cat(total_dist, dim=0)
        total_output = torch.cat(total_output, dim=0)
        if (epoch % 1 == 0) or (epoch > epochs - 10):
            R = torch.tensor(get_radius(total_dist, nu), device=device)
            test_auc, test_auprc = eval_ad_fedstar(model, test_loader, normal_class, c, R, device)
            print("loss train :", total_loss)
        losses_train.append(total_loss)
        aucs_test.append(test_auc)
        aucprs_test.append(test_auprc)

    return {'trainingLosses': losses_train, 'Aucs': aucs_test, 'Aucpr': aucprs_test}, R

def eval_ad_fedstar(model, test_loader, normal_class, c, R, device):
    model.eval()
    z, y = model.get_result(test_loader)
    z = torch.tensor(z).to(device)
    label_score = []
    dist = torch.sum((z - c) ** 2, dim=1)
    scores = dist - R ** 2
    labels = np.array(y.tolist())
    test_labels = np.where(labels == normal_class, 1, 0)
    labels = np.where(labels == normal_class, 0, 1)
    scores = np.array(scores.cpu().data.numpy().tolist())
    test_auprc = compute_auprc_baseline(test_labels, scores)
    test_auc = roc_auc_score(labels, scores)
    return test_auc, test_auprc