from torch.optim import AdamW, Adam

from .builder import OPTIMIZER

OPTIMIZER.register_module(module=Adam)
OPTIMIZER.register_module(module=AdamW)
