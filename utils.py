"""Utility functions for data loading
"""


def num_pieces(num, length, lower, upper):
    """Splitting dataset in (length)-unequal parts (lower = lower limit, upper = upper limit for every split)

    Args:
        num ([type]): [description]
        length ([type]): [description]
        lower ([type]): [description]
        upper ([type]): [description]

    Returns:
        [type]: [description]
    """
    ot = list(range(1, length + 1))[::-1]
    all_list = []
    for i in range(length - 1):
        n = random.randint(lower, min(upper, num - ot[i]))
        all_list.append(n)
        num -= n
    all_list.append(num)
    return all_list


def iid_clients(train_ds, n):
    split = utils.num_pieces(
        len(train_ds), n, 1000, 10000
    )  # Needs to be fraction of dataset size
    print("Dataset split: ", split)
    client_ds = random_split(train_ds, split)

    client_dls = [
        DataLoader(ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        for ds in client_ds
    ]
    # client_models = [MnistNet() for _ in range(n)]
    # client_optimizers = [Adam(model.parameters(), 0.001) for model in client_models]
    return client_dls


# Transmission function which takes bid as argument
def exponential_cutoff(bid):
    bid = torch.FloatTensor(bid)
    return 1 - np.exp(-1 * bid)