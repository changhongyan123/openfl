from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from openfl.experimental.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.runtime import LocalRuntime
from openfl.experimental.placement import aggregator, collaborator
import torchvision.transforms as transforms
import pickle
from pathlib import Path

import time
import os
import argparse
from cifar10_loader import CIFAR10
import warnings
import logging
warnings.filterwarnings("ignore")

# hyper-parameters for the optimizer
learning_rate = 0.005
momentum = 0.9


log_interval = 10

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


def default_optimizer(model, optimizer_type=None, optimizer_like=None):
    """
    Return a new optimizer based on the optimizer_type or the optimizer template
    """
    if optimizer_type == "SGD" or isinstance(optimizer_like, optim.SGD):
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_type == "Adam" or isinstance(optimizer_like, optim.Adam):
        return optim.Adam(model.parameters())


def FedAvg(models):
    new_model = models[0]
    if len(models) > 1:
        state_dicts = [model.state_dict() for model in models]
        state_dict = new_model.state_dict()
        for key in models[1].state_dict():
            state_dict[key] = np.sum(
                [state[key] for state in state_dicts], axis=0
            ) / len(models)
        new_model.load_state_dict(state_dict)
    return new_model


def inference(network, test_loader, device):
    network.eval()
    network.to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader)
    accuracy = float(correct / len(test_loader.dataset))
    print(
        (f"Test set: Avg. loss: {test_loss}, "
            f"Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * accuracy}%)")
    )
    network.to("cpu")
    return accuracy


def optimizer_to_device(optimizer, device):

    if optimizer.state_dict()["state"] != {}:
        if isinstance(optimizer, optim.SGD):
            for param in optimizer.param_groups[0]["params"]:
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad = param.grad.to(device)
        elif isinstance(optimizer, optim.Adam):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    else:
        raise (ValueError("No dict keys in optimizer state: please check"))


class FederatedFlow(FLSpec):
    def __init__(
        self,
        model,
        optimizers,
        device="cpu",
        total_rounds=10,
        top_model_accuracy=0,
        log_dir=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.global_model = Net()
        self.optimizers = optimizers
        self.total_rounds = total_rounds
        self.top_model_accuracy = top_model_accuracy
        self.device = device
        self.round_num = 0  # starting round
        self.log_dir = log_dir
        print(20 * "#")
        print(f"Round {self.round_num}...")
        print(20 * "#")

    @aggregator
    def start(self):
        self.start_time = time.time()
        print("Performing initialization for model")
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.next(
            self.aggregated_model_validation,
            foreach="collaborators",
            exclude=["private"],
        )

    # @collaborator  # Uncomment if you want ro run on CPU
    @collaborator(num_gpus=1)  # Assuming GPU(s) is available in the machine
    def aggregated_model_validation(self):
        print(
            ("Performing aggregated model validation for collaborator: "
                f"{self.input} in round {self.round_num}")
        )
        self.agg_validation_score = inference(self.model, self.test_loader, self.device)
        print(f"{self.input} value of {self.agg_validation_score}")
        self.collaborator_name = self.input
        self.next(self.train)

    # @collaborator  # Uncomment if you want ro run on CPU
    @collaborator(num_gpus=1)  # Assuming GPU(s) is available on the machine
    def train(self):
        print(20 * "#")
        print(
            f"Performing model training for collaborator {self.input} in round {self.round_num}"
        )

        self.model.to(self.device)
        self.optimizer = default_optimizer(
            self.model, optimizer_like=self.optimizers[self.input]
        )

        if self.round_num > 0:
            self.optimizer.load_state_dict(
                deepcopy(self.optimizers[self.input].state_dict())
            )
            optimizer_to_device(optimizer=self.optimizer, device=self.device)

          
        self.model.train()
        train_losses = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target).to(self.device)
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                train_losses.append(loss.item())

        self.loss = np.mean(train_losses)
        self.training_completed = True

        self.model.to("cpu")
        
        # save the model to the disk
        saving_path = f"{self.log_dir}/{self.input}_at_{self.round_num}.pkl"
        with open(saving_path, "wb") as handle:
            pickle.dump(self.model.state_dict(), handle)
        
        tmp_opt = deepcopy(self.optimizers[self.input])
        tmp_opt.load_state_dict(self.optimizer.state_dict())
        self.optimizer = tmp_opt
        torch.cuda.empty_cache()
        self.next(self.local_model_validation)

    # @collaborator  # Uncomment if you want ro run on CPU
    @collaborator(num_gpus=1)  # Assuming GPU(s) is available in the machine
    def local_model_validation(self):
        print(
            ("Performing local model validation for collaborator: "
                f"{self.input} in round {self.round_num}")
        )
        print(self.device)
        start_time = time.time()

        print("Test dataset performance")
        self.local_validation_score = inference(
            self.model, self.test_loader, self.device
        )
        print("Train dataset performance")
        self.local_validation_score_train = inference(
            self.model, self.train_loader, self.device
        )

        print(
            ("Doing local model validation for collaborator: "
                f"{self.input}: {self.local_validation_score}")
        )
        print(f"local validation time cost {(time.time() - start_time)}")


        self.next(self.join, exclude=["training_completed"])

    @aggregator
    def join(self, inputs):
        self.average_loss = sum(input.loss for input in inputs) / len(inputs)
        self.aggregated_model_accuracy = sum(
            input.agg_validation_score for input in inputs
        ) / len(inputs)
        self.local_model_accuracy = sum(
            input.local_validation_score for input in inputs
        ) / len(inputs)
        print(
            f"Average aggregated model validation values = {self.aggregated_model_accuracy}"
        )
        print(f"Average training loss = {self.average_loss}")
        print(f"Average local model validation values = {self.local_model_accuracy}")

        self.model = FedAvg([input.model.cpu() for input in inputs])
        self.global_model.load_state_dict(deepcopy(self.model.state_dict()))
        
        # save the model to the disk
        saving_path = f"{self.log_dir}/global_at_{self.round_num}.pkl"
        with open(saving_path, "wb") as handle:
            pickle.dump(self.model.state_dict(), handle)
            
        self.optimizers.update(
            {input.collaborator_name: input.optimizer for input in inputs}
        )

        del inputs
        self.next(self.check_round_completion)

    @aggregator
    def check_round_completion(self):
        if self.round_num != self.total_rounds:
            if self.aggregated_model_accuracy > self.top_model_accuracy:
                print(
                    ("Accuracy improved to "
                        f"{self.aggregated_model_accuracy} for round {self.round_num}")
                )
                self.top_model_accuracy = self.aggregated_model_accuracy
            self.round_num += 1
            print(20 * "#")
            print(f"Round {self.round_num}...")
            print(20 * "#")
            self.next(
                self.aggregated_model_validation,
                foreach="collaborators",
                exclude=["private"],
            )
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print(20 * "#")
        print("All rounds completed successfully")
        print(20 * "#")
        print("This is the end of the flow")
        print(20 * "#")


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
    
    argparser.add_argument(
        "--optimizer_type",
        type=str,
        default="SGD",
        help="Indicate optimizer to use for training",
    )
    argparser.add_argument(
        "--batch_size_train",
        type=int,
        default=32,
        help="Indicate the size of the training batches",
    )
    argparser.add_argument(
        "--comm_round",
        type=int,
        default=30,
        help="Indicate the communication rounds",
    )
    argparser.add_argument(
        "--batch_size_test",
        type=int,
        default=1000,
        help="Indicate the size of the test batches",
    )
    argparser.add_argument(
        "--split_idx",
        type=int,
        default=0,
        help="Training FL model based on which split",
    )
    argparser.add_argument(
        "--gpu_id",
        type=int,
        default=1,
        help="GPU index",
    )

    args = argparser.parse_args()
    # set the random seed for repeatable results
    random_seed = 10
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    
    # Setup participants
    aggregator = Aggregator()
    aggregator.private_attributes = {}

    # Setup collaborators with private attributes
    collaborator_names = [i for i in range(args.num_parties)]

    collaborators = [Collaborator(name=name) for name in collaborator_names]
    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{args.gpu_id}"
        ) 
    else:
        device = torch.device(
            "cpu"
        )

    all_data_file = f"{args.dataset}/data.pkl"
    data_partitioning = f"{args.dataset}/iid_{args.num_parties}_{args.train_dataset_ratio}_{args.test_dataset_ratio}_{args.random_seed}.pkl"
    if os.path.exists(all_data_file) and os.path.exists(data_partitioning):
        with open(all_data_file,"rb") as f:
            all_data = pickle.load(f)
        with open(data_partitioning,"rb") as f:
            list_dict = pickle.load(f)

    else:
        raise ValueError("data file does not exist")

    log_dir = f"results/iid_{args.num_parties}_{args.train_dataset_ratio}_{args.test_dataset_ratio}_{args.random_seed}/{args.split_idx}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    for idx, collab in enumerate(collaborators):
        train_index = list_dict[idx][args.split_idx]['train']
        test_index = list_dict[idx][args.split_idx]['train']

        local_train = deepcopy(all_data)
        local_test = deepcopy(all_data)

        local_train.data = local_train.data[train_index]
        local_train.targets = list(np.array(local_train.targets)[train_index])

        local_test.data = local_test.data[train_index]
        local_test.targets = list(np.array(local_test.targets)[train_index])

        logging.info(f"train size: {len(local_train)}, test size: {len(local_test)}")
    

        collab.private_attributes = {
            "train_loader": torch.utils.data.DataLoader(
                local_train, batch_size=args.batch_size_train, shuffle=True
            ),
            "test_loader": torch.utils.data.DataLoader(
                local_test, batch_size=args.batch_size_test, shuffle=False
            ),
            "log_dir": log_dir
        }

    # To activate the ray backend with parallel collaborator tasks run in their own process
    # and exclusive GPUs assigned to tasks, set LocalRuntime with backend='ray':
    local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators)#,backend='ray')

    print(f'Local runtime collaborators = {local_runtime.collaborators}')

    # change to the internal flow loop
    model = Net()
    top_model_accuracy = 0
    optimizers = {
        collaborator.name: default_optimizer(model, optimizer_type=args.optimizer_type)
        for collaborator in collaborators
    }
    flflow = FederatedFlow(
        model,
        optimizers,
        device,
        args.comm_round,
        top_model_accuracy,
        log_dir
    )

    flflow.runtime = local_runtime
    flflow.run()
