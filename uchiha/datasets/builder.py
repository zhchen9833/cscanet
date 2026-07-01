from torch.utils.data import DataLoader

from ..utils import get_root_logger
from ..utils import Registry

DATASET = Registry('dataset')

PIPELINES = Registry('pipelines')


def build_dataset(cfg, phase=None):
    """ build dataset based on configuration

    Args:
        cfg (dict): Configuration information, where the first key is type
        phase (str): train or val or test.

    Returns:
        Dataset: the built optimizer

    """
    logger = get_root_logger()
    logger.info(f"start building {phase} dataset...")

    dataset = DATASET.build(cfg)

    logger.info('success!')
    return dataset


def build_dataloader(dataset, cfg, phase=None):
    """ build dataloader based on configuration

    Args:
        dataset (Dataset): the dataset already built before
        cfg (dict): Configuration information, where the first key is type
        phase (str): train or val or test.

    Returns:
        Dataloader: the built dataloader

    """
    logger = get_root_logger()
    logger.info(f"start building {phase} dataloader...")

    dataloader = DataLoader(dataset, **cfg)
    logger.info('success!')

    return dataloader
