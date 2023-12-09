import argparse
import copy
import os
import random
from pathlib import Path

import torch

import setupGC
from training import *
import warnings
warnings.filterwarnings("ignore")


def process_selftrain(c, args, clients, server, local_epoch):
    print("Self-training ...")
    AUC, AUPRC = run_selftrain_GC(clients, server, local_epoch, args.normal_class, c, args.nu, args.data_group)
    return AUC, AUPRC


def process_fedavg(c, args, clients, server):
    print("\nDone setting up FedAvg devices.")
    print("Running FedAvg ...")
    AUC, AUPRC = run_fedavg(clients, server, args.num_rounds, args.local_epoch, args.normal_class, c, args.nu, args.data_group, samp = None)
    return AUC, AUPRC


def process_fedprox(c, args, clients, server, mu):
    print("\nDone setting up FedProx devices.")
    print("Running FedProx ...")
    AUC, AUPRC = run_fedprox(clients, server, args.num_rounds, args.local_epoch, mu, args.normal_class,  c, args.nu, args.data_group, samp=None)
    return AUC, AUPRC


def process_gcfl(c, args, clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL ...")
    AUC, AUPRC = run_gcfl(clients, server, args.num_rounds, args.local_epoch, args.normal_class,  c, args.nu, args.data_group, EPS_1, EPS_2)
    return AUC, AUPRC

def process_fedstar(c, args, clients, server):
    print("\nDone setting up FedStar devices.")
    print("Running FedStar ...")
    AUC, AUPRC = run_fedstar(args, clients, server, args.num_rounds, args.local_epoch, args.normal_class, c, args.nu, samp=None)
    return AUC, AUPRC


def process_fgad(clients, server, args):
    print("\nDone setting up FGAD devices.")
    print("Running FGAD ...")
    AUC, AUPRC = run_fgad(clients, server, args.num_rounds, args.local_epoch, args.pretrain_local_epoch,
                            args.normal_class, args.lamda, args.mu, args.data_group, samp=None)
    return AUC, AUPRC


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
    parser.add_argument('--num_repeat', type=int, default=2,
                        help='number of repeating rounds to simulate;')
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of GINconv layers')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=123)

    parser.add_argument('--datapath', type=str, default='./data',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs',
                        help='The base path for outputting.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='molecules')

    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--num_clients', help='number of clients',
                        type=int, default=10)
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--standardize', help='whether to standardize the distance matrix',
                        type=bool, default=False)
    parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=5)
    parser.add_argument('--epsilon1', help='the threshold epsilon1 for GCFL',
                        type=float, default=0.05)
    parser.add_argument('--epsilon2', help='the threshold epsilon2 for GCFL',
                        type=float, default=0.1)
    parser.add_argument('--temp', type=float, default=1.0, help='temperature of Knowledge Distillation')
    parser.add_argument('--pretrain_local_epoch', type=int, default=2,
                        help='number of pretrained local epochs;')
    parser.add_argument('--loss_mode', type=int, default=0)
    parser.add_argument('--normal_class', type=int, default=0, metavar='N',
                        help='normal class index')
    parser.add_argument('--lamda', type=float, default=0.1, metavar='N',
                        help='Weight of the perturbator loss')
    parser.add_argument('--mu', type=float, default=0.001, metavar='N',
                        help='Weight of the Knowledge Distillation loss')
    parser.add_argument('--algorithm', type=str, default='fgad', metavar='N',
                        help='optional algorithm, including self-train, fedavg, fedprox, gcfl, fedstar')
    parser.add_argument('--nu', dest='nu', type=float,
                        help='', default=0.001)
    parser.add_argument('--n_rw', type=int, default=16,
                        help='Size of position encoding (random walk).')
    parser.add_argument('--n_dg', type=int, default=16,
                        help='Size of position encoding (max degree).')
    parser.add_argument('--type_init', help='the type of positional initialization',
                        type=str, default='rw_dg', choices=['rw', 'dg', 'rw_dg', 'ones'])

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    seed_dataSplit = 123

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = "cpu"

    EPS_1 = args.epsilon1
    EPS_2 = args.epsilon2

    outbase = os.path.join(args.outbase, f'seqLen{args.seq_length}')
    if args.overlap and args.standardize:
        outpath = os.path.join(outbase, f"standardizedDTW/oneDS-overlap")
    elif args.overlap:
        outpath = os.path.join(outbase, f"oneDS-overlap")
    elif args.standardize:
        outpath = os.path.join(outbase, f"standardizedDTW/oneDS-nonOverlap")
    else:
        outpath = os.path.join(outbase, f"oneDS-nonOverlap")
    outpath = os.path.join(outpath, f'{args.data_group}-{args.num_clients}clients', f'eps_{EPS_1}_{EPS_2}')
    Path(outpath).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outpath}")

    """ distributed one dataset to multiple clients """

    if not args.convert_x:
        """ using original features """
        suffix = ""
        print("Preparing data (original features) ...")
    else:
        """ using node degree features """
        suffix = "_degrs"
        print("Preparing data (one-hot degree features) ...")

    splitedData, df_stats = setupGC.prepareData_multiDS(args.datapath, args.normal_class, args.data_group,
                                                        args.batch_size, convert_x=args.convert_x, seed=seed_dataSplit)
    print("Done")
    repNum = args.num_repeat
    hist_AUC = []
    hist_AUCPR = []
    for epoch in range(repNum):
        if args.algorithm == 'fedavg':
            c = torch.randn(args.hidden * args.nlayer).to(args.device)
            init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
            AUC, AUPRC = process_fedavg(c, args, clients=copy.deepcopy(init_clients),
                                        server=copy.deepcopy(init_server))
        elif args.algorithm == 'fedprox':
            c = torch.randn(args.hidden * args.nlayer).to(args.device)
            init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
            AUC, AUPRC = process_fedprox(c, args, clients=copy.deepcopy(init_clients),
                                         server=copy.deepcopy(init_server), mu=0.01)
        elif args.algorithm == 'gcfl':
            c = torch.randn(args.hidden * args.nlayer).to(args.device)
            init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
            AUC, AUPRC = process_gcfl(c, args, clients=copy.deepcopy(init_clients),
                                      server=copy.deepcopy(init_server))
        elif args.algorithm == 'self_train':
            c = torch.randn(args.hidden * args.nlayer).to(args.device)
            init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
            AUC, AUPRC = process_selftrain(c, args, clients=copy.deepcopy(init_clients),
                                           server=copy.deepcopy(init_server), local_epoch=100)
        elif args.algorithm == 'fedstar':
            args.n_se = args.n_rw + args.n_dg
            c = torch.randn(args.hidden).to(args.device)
            init_clients, init_server, init_idx_clients = setupGC.setup_devices_fedstar(splitedData, args)
            AUC, AUPRC = process_fedstar(c, args, clients=copy.deepcopy(init_clients),
                                         server=copy.deepcopy(init_server))
        elif args.algorithm == 'fgad':
            init_clients, init_server, init_idx_clients = setupGC.setup_devices_fgad(splitedData, args)
            AUC, AUPRC = process_fgad(clients=copy.deepcopy(init_clients),
                                      server=copy.deepcopy(init_server), args=args)
        hist_AUC.append(AUC)
        hist_AUCPR.append(AUPRC)
    Mean_AUC = np.around([np.mean(np.array(hist_AUC)), np.std(np.array(hist_AUC))], decimals=4)
    Mean_AUPRC = np.around([np.mean(np.array(hist_AUCPR)), np.std(np.array(hist_AUCPR))], decimals=4)
    with open('./result_multi/' + args.algorithm + '_' + args.data_group + '_result.txt', 'a') as f:
        f.write('Average AUC:' + str(Mean_AUC[0] * 100) + '$\pm$' + str(Mean_AUC[1] * 100) + '\n')
        f.write('Average AUCPR:' + str(Mean_AUPRC[0] * 100) + '$\pm$' + str(Mean_AUPRC[1] * 100) + ')\n')
