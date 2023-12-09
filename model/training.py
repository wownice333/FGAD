import numpy as np
import pandas as pd
import scipy
import torch
from Utils.utils import obtain_avg_result, init_metric
import Utils.utils

def run_selftrain(clients, server, local_epoch, normal_class, c, nu, DS):
    # all clients are initialized with the same weights
    for client in clients:
        client.download_from_server(server)
    client_AUC = []
    client_AUCPR = []
    for client in clients:
        client.local_train(local_epoch, normal_class, c, nu)
    for client in clients:
        test_auc, test_auprc = client.evaluate()
        client_AUC.append(test_auc)
        client_AUCPR.append(test_auprc)
    avg_AUC, avg_AUPRC = obtain_avg_result(client_AUC, client_AUCPR, DS, 'self-train')
    return avg_AUC, avg_AUPRC


def run_fedavg(clients, server, COMMUNICATION_ROUNDS, local_epoch, normal_class, c, nu, DS, samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    init_metric()
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            # only get weights of graphconv layers
            client.local_train(local_epoch, normal_class, c, nu)

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(server)
        client_AUC = []
        client_AUCPR = []
        for client in clients:
            test_auc, test_auprc = client.evaluate()
            client_AUC.append(test_auc)
            client_AUCPR.append(test_auprc)
        avg_AUC, avg_AUPRC = obtain_avg_result(client_AUC, client_AUCPR, DS, 'FedAvg')
    return avg_AUC, avg_AUPRC


def run_fedprox(clients, server, COMMUNICATION_ROUNDS, local_epoch, mu, normal_class, c, nu, DS,samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    if samp == 'random':
        sampling_fn = server.randomSample_clients
    init_metric()
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            client.local_train_prox(local_epoch, mu, normal_class, c, nu)

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(server)
            # cache the aggregated weights for next round
            client.cache_weights()

        client_AUC = []
        client_AUCPR = []
        for client in clients:
            test_auc, test_auprc = client.evaluate()
            client_AUC.append(test_auc)
            client_AUCPR.append(test_auprc)
        avg_AUC, avg_AUPRC = obtain_avg_result(client_AUC, client_AUCPR, DS, 'FedProx')
    return avg_AUC, avg_AUPRC


def run_gcfl(clients, server, COMMUNICATION_ROUNDS, local_epoch, normal_class, c, nu, DS, EPS_1, EPS_2):
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    # tmp = -1.
    init_metric()
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(server)

        participating_clients = server.randomSample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.compute_weight_update(local_epoch,normal_class, c, nu)
            client.reset()

        similarities = server.compute_pairwise_similarities(clients)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20:
                server.cache_model(idc, clients[idc[0]].W)
                c1, c2 = server.min_cut(similarities[idc][:, idc], idc)
                cluster_indices_new += [c1, c2]
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]  # initial: [[0, 1, ...]]

        server.aggregate_clusterwise(client_clusters)

        client_AUC = []
        client_AUCPR = []
        for client in clients:
            test_auc, test_auprc = client.evaluate()
            client_AUC.append(test_auc)
            client_AUCPR.append(test_auprc)
        avg_AUC, avg_AUPRC = obtain_avg_result(client_AUC, client_AUCPR, DS, 'GCFL')
    return avg_AUC, avg_AUPRC

def run_fedstar(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, normal_class, c, nu, samp=None, frac=1.0, summary_writer=None):
    for client in clients:
        client.download_from_server_fedstar(server)
    DS = args.data_group
    init_metric()
    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    tmp = -1.
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            # only get weights of graphconv layers
            client.local_train_fedstar(local_epoch, normal_class, c, nu)

        server.aggregate_weights_se(selected_clients)
        for client in selected_clients:
            client.download_from_server_fedstar(server)

        client_AUC = []
        client_AUCPR = []
        for client in clients:
            test_auc, test_auprc = client.evaluate()
            client_AUC.append(test_auc)
            client_AUCPR.append(test_auprc)
        avg_AUC, avg_AUPRC = obtain_avg_result(client_AUC, client_AUCPR, DS, 'FedStar')
    return avg_AUC, avg_AUPRC

def run_fgad(clients, server, COMMUNICATION_ROUNDS, local_epoch, pretrain_local_epoch, normal_class, lamda, mu, DS,
               samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    # print(Utils.utils.hist_auc)
    # print(Utils.utils.hist_auprc)
    init_metric()
    # print('after:',Utils.utils.hist_auc)
    # print(Utils.utils.hist_auprc)
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        print('conmmunication rounds:', c_round)
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
            pretrain = True
        else:
            selected_clients = sampling_fn(clients, frac)
            pretrain = False

        for client in selected_clients:
            # print('Client ', client)
            # only get weights of graphconv layers
            client.local_train(local_epoch, pretrain, pretrain_local_epoch, normal_class, lamda, mu)

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(server)
        client_AUC=[]
        client_AUCPR = []
        for client in clients:
            test_auc, test_auprc = client.evaluate()
            client_AUC.append(test_auc)
            client_AUCPR.append(test_auprc)
        
        avg_AUC, avg_AUPRC = obtain_avg_result(client_AUC, client_AUCPR, DS, 'FGAD')
        if avg_AUC>0.64:
            idx=0
            for client in clients:
                client.save(idx, avg_AUC)
                idx+=1
    return avg_AUC, avg_AUPRC