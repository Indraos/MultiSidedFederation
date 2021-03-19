from torch.utils.data import random_split, DataLoader


def biased_partition(num, total, lower, upper)
    """Generating a biased integer partition

    Args:
        num (int): length of the integer partition
        total (int): sum of the integer partition
        lower (int): lower bound on summands
        upper (int): upper bound on summands

    Returns:
        [int]: Partition
    """
    backward_list = list(range(1, total + 1))[::-1]
    partition = []
    for i in range(length - 1):
        n = random.randint(lower, min(upper, num - backward_list[i]))
        partition.append(n)
        num -= n
    all_list.append(num)
    return all_list


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
    partition = num_pieces(len(dataset), n, lower_lmt, upper_lmt)
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
    bid = torch.FloatTensor(bid)
    return 1 - np.exp(-1 * bid)