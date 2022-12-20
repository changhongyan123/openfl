from copy import deepcopy
import torch
import numpy as np
import torchvision.transforms as transforms
import pickle
from pathlib import Path
import os
import argparse
# from cifar10_loader import CIFAR10
from torchvision import datasets

import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)



def get_iid_index(all_index, num_parties,num_splits,train_ratio,test_ratio):
    """
    This function is used for generate the splits of the dataset;
    For each collaboator, we have a list of splits. For each split, we have the train data, test data and the rest of the dataset.
    There is no overlapping between the datasets from collaboators
    """
    np.random.shuffle(all_index)
    index_per_party = [all_index[idx::num_parties] for idx in range(num_parties)]
    list_dict = {}
    
    for party in range(num_parties):
        party_index = index_per_party[party]
        num_total = len(party_index)
        split_list = []
        for split_idx in range(num_splits):
            splits = {}
            selected_index = np.random.choice(party_index, int((train_ratio+test_ratio)*num_total),replace=False)
            splits['train'] = selected_index[:int(train_ratio*num_total)]
            splits['test'] = selected_index[int(train_ratio*num_total):int((train_ratio+test_ratio)*num_total)]
            splits['rest'] = np.array([i for i in party_index if i not in selected_index])

            split_list.append(splits)
        list_dict[party] = split_list
    return list_dict


def get_richlet_index(all_index, Y, num_parties,num_splits,train_ratio,test_ratio,dirichlet_alpha):
    idx_batch = [[] for _ in range(num_parties)]
    N_total_samples = len(all_index)
    num_classes = len(np.unique(Y))
    for k in range(num_classes):
        idx_k = np.where(Y==k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(dirichlet_alpha,num_parties))
        proportions = np.array([
            p * (len(idx_j) < N_total_samples / num_parties) 
            for p, idx_j in zip(proportions,idx_batch)
        ])
        proportions = proportions/ proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        index_per_party = [
            idx_j + idx.tolist()
            for idx_j, idx in zip(idx_batch, np.split(idx_k,proportions))
        ]
    list_dict = {}
    
    for party in range(num_parties):
        party_index = index_per_party[party]
        num_total = len(party_index)
        split_list = []
        for split_idx in range(num_splits):
            splits = {}
            selected_index = np.random.choice(party_index, int((train_ratio+test_ratio)*num_total),replace=False)
            splits['train'] = selected_index[:int(train_ratio*num_total)]
            splits['test'] = selected_index[int(train_ratio*num_total):int((train_ratio+test_ratio)*num_total)]
            splits['rest'] = np.array([i for i in party_index if i not in selected_index])
            split_list.append(splits)
        list_dict[party] = split_list
    return list_dict




if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Which dataset to use",
    )
    argparser.add_argument(
        "--num_splits",
        type=int,
        default=50,
        help="Indicate how many splits we want",
    )
    argparser.add_argument(
        "--test_dataset_ratio",
        type=float,
        default=0.4,
        help="Indicate the what fraction of the sample will be used for testing",
    )
    argparser.add_argument(
        "--train_dataset_ratio",
        type=float,
        default=0.4,
        help="Indicate the what fraction of the sample will be used for training",
    )
    
    argparser.add_argument(
        "--num_parties",
        type=int,
        default=10,
        help="Indicate the how many collaboators",
    )
    argparser.add_argument(
        "--random_seed", type=int, default=0, help="Indicate random seed"
    )
    argparser.add_argument(
        "--partitioning",
        type=str,
        default="iid",
        help="Indicate how to split the dataset among parties",
    )
    argparser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=0.5,
        help="Indicate the parameter for the dirichlet partitioning",
    )
   
    
    args = argparser.parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    
    if args.dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])

        cifar_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

        cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

        N_total_samples = len(cifar_test) + len(cifar_train)
        total_index = np.arange(N_total_samples)
        X = np.concatenate([cifar_test.data, cifar_train.data])
        Y = np.concatenate([cifar_test.targets, cifar_train.targets]).tolist()
        
        
        if os.path.exists(f'{args.dataset}/data.pkl') is False:
            all_data = deepcopy(cifar_train)
            all_data.data = X
            all_data.targets = Y
            Path(f"{args.dataset}").mkdir(parents=True,exist_ok=True)
            
            with open(f"{args.dataset}/data.pkl","wb") as f:
                pickle.dump(all_data,f)
              
              
    # get the list of index for the experiments
    if args.partitioning == "iid":
        sample_list_dict = get_iid_index(total_index, args.num_parties,args.num_splits,args.train_dataset_ratio,args.test_dataset_ratio)
        save_name = f"{args.dataset}/iid_{args.num_parties}_{args.train_dataset_ratio}_{args.test_dataset_ratio}_{args.random_seed}"
    elif args.partitioning == "dirichlet":
        sample_list_dict = get_richlet_index(total_index, Y, args.num_parties,args.num_splits,args.train_dataset_ratio,args.test_dataset_ratio,args.dirichlet_alpha)
        save_name = f"{args.dataset}/dirichlet_{args.dirichlet_alpha}_{args.num_parties}_{args.train_dataset_ratio}_{args.test_dataset_ratio}_{args.random_seed}"

    if os.path.exists(f"{save_name}.pkl"):
        logging.error(f"the {save_name}.pkl exists, please make sure you want to change it")
    else:
        with open(f"{save_name}.pkl","wb") as f:
            pickle.dump(sample_list_dict,f)
        logging.info(f"dataset split is saved into {save_name}")

