from torch.utils.data import random_split, DataLoader
import numpy as np


def biased_partition(sum, num, lower, upper):
    """Generating a biased integer partition

    Args:
        num (int): length of the integer partition
        sum (int): sum of the integer partition
        lower (int): lower bound on summands
        upper (int): upper bound on summands

    Returns:
        [int]: Partition
    """
    backward_list = list(range(1, num + 1))[::-1]
    partition = []
    for i in range(num - 1):
        element = np.random.randint(lower, min(upper, sum - backward_list[i]))
        partition.append(element)
        sum -= element
    partition.append(sum)
    print(partition)
    return partition


def iid_clients(dataset, n, lower_lmt, upper_lmt, batch_size):
    """Generate list of iid clients

    Args:
        dataset (np.array): training set
        total (int): Size of training set
        lower (int): Lower limit on training set size
        upper (int): Upper limit on training set size
        batch_size (int): Batch size for DataLoader

    Returns:
        [Data]: Datasets
    """
    partition = biased_partition(len(dataset), n, lower_lmt, upper_lmt)
    client_ds = random_split(dataset, partition)
    client_dls = [
        DataLoader(ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        for ds in client_ds
    ]
    return client_dls


def exponential_cutoff(bid):
    """Likelihood of receiving each model for evaluation given bid

    Args:
        bid (float): bid of an agent

    Returns:
        float: likelihood of receiving a model
    """
    return 1 - np.exp(-1 * bid)