import numpy as np
import pandas as pd
import scipy
import torch
from Utils.utils import obtain_avg_result, init_metric

def run_fgad(clients, server, COMMUNICATION_ROUNDS, local_epoch, pretrain_local_epoch, normal_class, lamda, mu, DS,
               samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    init_metric()
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
    return avg_AUC, avg_AUPRC