import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from .datasets import RealFakeDataset

    

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt, preprocess=None):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = RealFakeDataset(opt)
    batch_size = int(opt.batch_size / max(1, len(opt.gpu_ids)))
    if '2b' in opt.arch:
        dataset.transform = preprocess
    if opt.class_bal and len(opt.gpu_ids) <= 1:
        sampler = get_bal_sampler(dataset)
    elif len(opt.gpu_ids) > 1:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    data_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False if isinstance(sampler, DistributedSampler) else shuffle,
                            sampler=sampler,
                            num_workers=int(opt.num_threads),
                            worker_init_fn=loader_worker_init_fn(dataset))
    return data_loader


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    if hasattr(loader, "sampler"):
        sampler = loader.sampler
    else:
        raise RuntimeError(
            "Unknown sampler for IterableDataset when shuffling dataset"
        )
    assert isinstance(
        sampler, (WeightedRandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)


def loader_worker_init_fn(dataset):
    """
    Create init function passed to pytorch data loader.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
    """
    return None