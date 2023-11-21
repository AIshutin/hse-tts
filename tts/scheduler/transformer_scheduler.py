from typing import Any
import torch


class TransformerLambda:
    def __init__(self, alpha, d_model, warmup_steps) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.alpha = alpha

    def __call__(self, step) -> Any:
        step += 1
        return self.alpha * (self.d_model ** -0.5) \
               * min(step ** -0.5, step * self.warmup_steps ** -1.5)


def getTransformerScheduler(optimizer, alpha, d_model, warmup_steps):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, TransformerLambda(alpha, d_model, warmup_steps), 
                                             last_epoch=-1)