import random
import time
import math
from datetime import timedelta
import numpy as np
import torch
from omegaconf import OmegaConf

goodness_functions = {
    "sum_of_squares":  lambda h: torch.sum(h ** 2, dim=-1),
    "log_sum_exp":      lambda h: torch.logsumexp(h, dim=-1),
    "sparse_l1":       lambda h: torch.sum(h ** 2, dim=-1) - 0.1 * torch.sum(torch.abs(h), dim=-1),
    "huber_norm":      lambda h: torch.sum(torch.where(torch.abs(h) <= 20.0, 0.5 * (h ** 2), 1.0 * (torch.abs(h) - 0.5)), dim=-1),
    "tempered_energy": lambda h: torch.sum(torch.exp((h ** 2) / 20.0), dim=-1),
    "outlier_trimmed": lambda h: torch.sum(torch.topk(h ** 2, k=int(h.shape[-1] * 0.9), dim=-1, largest=False).values, dim=-1),
    "ojas_rule":       lambda h: torch.sum(h ** 2, dim=-1) - 0.1 * torch.sum(h ** 4, dim=-1),
}

def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    print(OmegaConf.to_yaml(opt))
    return opt

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def dict_to_cuda(d):
    for key, value in d.items():
        d[key] = value.cuda(non_blocking=True)
    return d

def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels

def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr

def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(opt, epoch, opt.training.learning_rate)
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(opt, epoch, opt.training.downstream_learning_rate)
    return optimizer

def get_accuracy(opt, output, target):
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size

def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")
    print(f"{partition} \t \tTime: {timedelta(seconds=iteration_time)} \t", end="")
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()

def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict
