from ..utils import Registry, get_root_logger

OPTIMIZER = Registry('optimizer')

CRITERION = Registry('criterion')

SCHEDULER = Registry('scheduler')


def build_optimizer(params, cfg):
    """ build optimizer based on configuration

    Args:
        params (generator): parameters of the model
        cfg (dict): Configuration information, where the first key is type

    Returns:
        returns the built optimizer

    """
    logger = get_root_logger()
    logger.info("start building optimizer...")

    optimizer = OPTIMIZER.get(cfg.pop('type'))
    optimizer = optimizer(params, **cfg)

    logger.info('success!')
    return optimizer


def build_criterion(cfg):
    """ build loss function based on configuration

    Args:
        cfg (dict): Configuration information, where the first key is type

    Returns:
        returns the built loss function

    """
    logger = get_root_logger()
    logger.info("start building criterion...")

    criterion = CRITERION.build(cfg)

    logger.info('success!')
    return criterion


def build_scheduler(optimizer, cfg):
    """ build lr scheduler based on configuration

    Args:
        optimizer (class): the optimizer already built before
        cfg (dict): Configuration information, where the first key is type

    Returns:
        returns the built scheduler

    """
    scheduler_cfg = cfg.copy()
    logger = get_root_logger()
    logger.info("start building scheduler...")

    scheduler = SCHEDULER.get(scheduler_cfg.pop('type'))
    scheduler = scheduler(optimizer, **scheduler_cfg)

    logger.info('success!')
    return scheduler
