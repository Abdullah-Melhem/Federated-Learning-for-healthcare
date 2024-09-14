import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST


def get_mnist(data_path):
    """Download MNIST and apply minimal transformation."""

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset


#  num_partitions is the number of clients
def prepare_dataset(num_partitions: int = 10, batch_size: int = 5, val_ratio: float = 0.1, data_path: str = "./data"):
    """Download MNIST and generate IID partitions."""

    # download MNIST in case it's not already in the system
    trainset, testset = get_mnist(data_path)

    # split trainset into `num_partitions` trainsets (one per client)
    # figure out number of training examples per partition
    num_images = len(trainset) // num_partitions
   
    partition_len = [num_images] * num_partitions
    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )



    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    # for each train set, let's put aside some training examples for validation
    for trainset_ in trainsets:
        num_total = len(trainset_) # in this case it should be 6K images
        num_val = int(val_ratio * num_total) # eval_ portion
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        # construct data loaders and append to their respective list.
        # In this way, the i-th client will get the i-th element in the trainloaders list and the i-th element in the valloaders list
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

   
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader
