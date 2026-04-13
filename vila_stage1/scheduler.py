import functools

import torch


def _linear_decay_warmup(iteration, warmup_iterations, total_iterations):
    if iteration < warmup_iterations:
        return iteration / warmup_iterations
    return 1.0 - ((iteration - warmup_iterations) / (total_iterations - warmup_iterations))


def linear_annealing_with_warmup(optimizer, warmup_steps, total_steps):
    decay_func = functools.partial(
        _linear_decay_warmup,
        warmup_iterations=warmup_steps,
        total_iterations=total_steps,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, decay_func)
