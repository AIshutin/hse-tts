from typing import Any
import torch


class TransformerLambda:
    def __init__(self, d_model, warmup_steps) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step) -> Any:
        step += 1
        lr = (self.d_model ** -0.5) \
            * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        return lr


def getTransformerScheduler(optimizer, d_model, warmup_steps, **kwargs):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, TransformerLambda(d_model, warmup_steps), 
                                             last_epoch=-1)