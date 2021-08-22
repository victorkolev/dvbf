import json
import logging
import os
import random
import sys

import torch


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grad_norm(net):
    total_norm = 0
    for p in net.parameters():
        if p.grad is not None:
            para_norm = p.grad.data.norm(2)
            total_norm += para_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def create_save_dir(name):
    path = os.path.join("results", name)
    create_folders_if_necessary(path)
    return path


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger()


def save_args(model_dir, args):
    path = os.path.join(model_dir, "params.json")
    with open(path, "w") as f:
        json.dump(vars(args), f)


def get_model_state(model_dir):
    path = os.path.join(model_dir, "model.pt")
    if os.path.exists(path):
        with open(path, "r") as f:
            return torch.load(f)

    return None


def save_model_state(model_dir, net):
    path = os.path.join(model_dir, "model.pt")
    torch.save(net.state_dict(), path)
