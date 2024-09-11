import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from Utils.compute_auprc import compute_auprc

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from Utils.initialize import *
from Utils.losses import *



class Client_GC():
    def __init__(self, shared_GIN, sd_model, tea_model, stu_model, client_id, client_name, train_size, dataLoader,
                 optimizer, args):
        self.shared_GIN = shared_GIN.to(args.device)
        self.tea_model = tea_model.to(args.device)
        self.stu_model = stu_model.to(args.device)
        self.sd_model = sd_model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args

        self.W_stu = {key: value for key, value in self.stu_model.named_parameters()}
        self.dW_stu = {key: torch.zeros_like(value) for key, value in self.stu_model.named_parameters()}
        self.W_old_stu = {key: value.data.clone() for key, value in self.stu_model.named_parameters()}

        self.gconvNames_stu = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

    def download_from_server(self, server):
        self.gconvNames = server.W.keys()
        for k in server.W:
            self.W_stu[k].data = server.W[k].data.clone()

    def cache_weights(self):
        for name in self.W_stu.keys():
            self.W_old_stu[name].data = self.W_stu[name].data.clone()

    def reset(self):
        copy(target=self.W_stu, source=self.W_old_stu, keys=self.gconvNames)

    def local_train(self, local_epoch, pretrain, pretrain_local_epoch, normal_class, lamda, mu):
        """ For FGAD """
        train_stats = train_gc(self.shared_GIN, self.stu_model, self.tea_model, self.sd_model, self.dataLoader,
                               self.optimizer, self.args.temp, pretrain_local_epoch, local_epoch, normal_class,
                               self.args.device,
                               pretrain, lamda, mu, self.args.loss_mode)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W_stu)).item()

        weights_conv = {key: self.W_stu[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W_stu.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W_stu[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def compute_weight_update(self, pretrain_local_epoch, local_epoch):
        """ For FGAD """
        copy(target=self.W_old_stu, source=self.W_stu, keys=self.gconvNames)

        train_stats = train_gc(self.stu_model, self.tea_model, self.sd_model, self.dataLoader, self.optimizer,
                               self.args.temp, pretrain_local_epoch, local_epoch, self.args.device, self.args.loss_mode)
        subtract_(target=self.dW_stu, minuend=self.W_stu, subtrahend=self.W_old_stu)

        self.train_stats = train_stats

        self.weightsNorm = torch.norm(flatten(self.W_stu)).item()

        weights_conv = {key: self.W_stu[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        dWs_conv = {key: self.dW_stu[key] for key in self.gconvNames}
        self.convDWsNorm = torch.norm(flatten(dWs_conv)).item()

        grads = {key: value.grad for key, value in self.W_stu.items()}
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W_stu[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    def evaluate(self):
        return eval_gc(self.shared_GIN, self.stu_model, self.dataLoader['test'], self.args.normal_class,
                       self.args.device)
    def save(self, idx, avg_AUC):
        self.shared_GIN.eval()
        self.stu_model.eval()
        torch.save({'shared_GIN': self.shared_GIN.state_dict(), 'stu_head':self.stu_model.state_dict()}, './weight/'+self.args.data_group+'/Client_' + str(idx) + '_' + str(avg_AUC) + '.pth')


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



def train_gc(shared_encoder, model_stu, model_tea, sd_model, dataloaders, optimizer, temp, pretrain_local_epoch,
             local_epoch, normal_class, device, pretrain, lamda, mu, loss_mode=0):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    aucs_test = []
    aucprs_test = []
    f1s_test = []
    recalls_test = []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    if pretrain == True:
        for epoch in range(pretrain_local_epoch):
            model_tea.train()
            sd_model.train()
            total_loss = 0.
            ngraphs = 0
            acc_sum = 0
            for _, batch in enumerate(train_loader):
                data = batch.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if data.x is None or len(data.x.shape) == 1 or data.x.shape[1] == 0:
                    adj = torch.zeros(data.batch.shape[0], data.batch.shape[0])
                else:
                    adj = torch.zeros(data.x.shape[0], data.x.shape[0])
                # Restore adj from edge_index
                adj[data.edge_index] = 1
                adj = adj.to(device)

                optimizer.zero_grad()
                tea_z, _ = shared_encoder(x, adj, batch)
                pred = model_tea(tea_z).squeeze()
                label = data.y
                target = torch.ones_like(pred).float()

                target[:, 0] = 0
                target_fake = torch.ones_like(pred).float()
                target_fake[:, 1] = 0

                # generate fake samples
                fake_A = sd_model(x, edge_index, batch)
                tea_z_neg, _ = shared_encoder(x, fake_A, batch)
                fake_pred = model_tea(tea_z_neg).squeeze()
                loss = F.cross_entropy(pred, target) + F.cross_entropy(fake_pred,
                                                                       target_fake) + lamda * sd_model.loss(fake_A,
                                                                                                            adj,
                                                                                                            sd_model.mean,
                                                                                                            sd_model.logstd)
                loss.backward()
                optimizer.step()
    else:
        pass
    for epoch in range(local_epoch):
        shared_encoder.train()
        model_stu.train()
        model_tea.train()
        sd_model.train()
        total_loss = 0.
        ngraphs = 0
        acc_sum = 0
        for _, batch in enumerate(train_loader):
            data = batch.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch

            if data.x is None or len(data.x.shape) == 1 or data.x.shape[1] == 0:
                adj = torch.zeros(data.batch.shape[0], data.batch.shape[0])
            else:
                adj = torch.zeros(data.x.shape[0], data.x.shape[0])
            # Restore adj from edge_index
            adj[data.edge_index] = 1
            adj = adj.to(device)

            optimizer.zero_grad()
            z, _ = shared_encoder(x, adj, batch)
            pred = model_tea(z).squeeze()
            pred_stu = model_stu(z).squeeze()
            label = data.y
            target = torch.ones_like(pred).float()
            target[:, 0] = 0
            target_fake = torch.ones_like(pred).float()
            target_fake[:, 1] = 0

            # generate fake samples
            fake_A = sd_model(x, edge_index, batch)
            tea_z_neg, _ = shared_encoder(x, fake_A, batch)
            fake_pred = model_tea(tea_z_neg).squeeze()
            loss = F.cross_entropy(pred, target) + F.cross_entropy(fake_pred,
                                                                   target_fake) + F.cross_entropy(
                pred_stu, target) + lamda * sd_model.loss(fake_A, adj, sd_model.mean,
                                                          sd_model.logstd) + mu * com_distillation_loss(
                pred, pred_stu, temp, loss_mode)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            ngraphs += data.num_graphs

        total_loss /= ngraphs
        # test_auc, test_auprc = eval_gc(shared_encoder,
                                                                                                #   model_stu,
                                                                                                #   test_loader,
                                                                                                #   normal_class, device)
        # print("loss train :", total_loss)
        # print("loss train :", total_loss, "auc test:", test_auc, "aucpr test:", test_auprc)

        # losses_train.append(total_loss)
        # aucs_test.append(test_auc)
        # aucprs_test.append(test_auprc)
    return {'trainingLosses': losses_train, 'Aucs': aucs_test, 'Aucpr': aucprs_test}


def eval_gc(shared_encoder, model, test_loader, normal_class, device):
    shared_encoder.eval()
    model.eval()
    total_loss = 0.
    ngraphs = 0
    label_score = []
    for batch in test_loader:
        data = batch.to(device)
        if data.x is None or len(data.x.shape) == 1 or data.x.shape[1] == 0:
            adj = torch.zeros(data.batch.shape[0], data.batch.shape[0])
        else:
            adj = torch.zeros(data.x.shape[0], data.x.shape[0])
        # Restore adj from edge_index
        adj[data.edge_index] = 1
        adj = adj.to(device)

        with torch.no_grad():
            z, _ = shared_encoder(batch.x, adj, batch.batch)
            pred = model(z)
            sigmoid_scores = torch.softmax(pred, dim=1)
            pred = pred[:, 1]
            sigmoid_scores = sigmoid_scores[:, 1]
            label = batch.y.float()
        label_score += list(zip(label.cpu().data.numpy().tolist(),
                                pred.cpu().data.numpy().tolist(),
                                sigmoid_scores.cpu().data.numpy().tolist(),
                                z.cpu().data.numpy().tolist()))

        ngraphs += batch.num_graphs
    labels, scores, sigmoid_scores, middle = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    sigmoid_scores = np.array(sigmoid_scores)
    middle = np.array(middle)
    if isinstance(normal_class, list):
        labels = labels
        normal_idx = []
        sum_idx = list(range(len(labels)))
        for nc in range(len(normal_class)):
            normal_idx.extend(np.where(labels == normal_class[nc])[0])
        labels[normal_idx] = 1
        retain_idx = list(set(sum_idx).difference(normal_idx))
        labels[retain_idx] = 0
    else:
        labels = np.where(labels == normal_class, 1, 0)
    test_auprc= compute_auprc(labels, scores)
    test_auc = roc_auc_score(labels, scores)
    return test_auc, test_auprc


# KL-divergence Loss for knowledge distillation
def com_distillation_loss(t, s, temp, loss_mode):
    s_dist = F.log_softmax(s / temp, dim=-1)
    t_dist = F.softmax(t / temp, dim=-1)
    if loss_mode == 0:
        kd_loss = temp * temp * F.kl_div(s_dist, t_dist)
    elif loss_mode == 1:
        kd_loss = temp * temp * F.kl_div(s_dist, t_dist.detach())

    return kd_loss
