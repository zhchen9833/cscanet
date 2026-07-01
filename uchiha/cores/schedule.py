import math
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, _LRScheduler

from .builder import SCHEDULER

SCHEDULER.register_module(module=StepLR)
SCHEDULER.register_module(module=CosineAnnealingLR)


@SCHEDULER.register_module()
class LinearWarmupCosineLR(_LRScheduler):
    """
    Linear preheating cosine annealing learning rate scheduler.

    Args:
        optimizer: optimizer object
        total_epochs: total training rounds
        warmup_epochs: Number of warm-up rounds (5% of the total number of rounds by default)
        warmup_start_lr: initial learning rate default 1 of initial lr
        min_lr: minimum learning rate 1 of default initial lr
        last_epoch: parameters used when resuming training
    """

    def __init__(self,
                 optimizer,
                 total_epochs,
                 warmup_epochs=None,
                 warmup_start_lr=None,
                 min_lr=None,
                 last_epoch=-1):

        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs if warmup_epochs is not None else max(1, int(total_epochs * 0.05))
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr is not None else optimizer.defaults['lr'] * 0.01
        self.min_lr = min_lr if min_lr is not None else optimizer.defaults['lr'] * 0.01

        # increment of linear warm up
        self.warmup_delta = (optimizer.defaults['lr'] - self.warmup_start_lr) / self.warmup_epochs

        # parameters of cosine annealing
        self.cosine_epochs = max(1, total_epochs - self.warmup_epochs)

        super(LinearWarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # linear warm up stage
            curr_lr = self.warmup_start_lr + self.last_epoch * self.warmup_delta
            return [curr_lr for _ in self.base_lrs]

        else:
            # cosine annealing stage
            cosine_epoch = self.last_epoch - self.warmup_epochs
            progress = cosine_epoch / self.cosine_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            curr_lr = self.min_lr + (self.base_lrs[0] - self.min_lr) * cosine_decay
            return [curr_lr for _ in self.base_lrs]

    def get_last_lr(self):
        """ current learning rate """
        return self.get_lr()
