import numpy as np
import random
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image  # Save a given Tensor into an image file.
from torch.utils.data import DataLoader
import os
import cv2
import logging


def to_device(device, *args):
    if not isinstance(device, torch.device):
        raise Exception('The first parameter is not a valid torch device instance')
    for arg in args:
        arg.to(device)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# give an experiment name and make an experiment directory
def exp_init(exp_name, dataset, network, generator):
    exp_name = exp_name + dataset + "_" + network + "_" 
    exp_name = exp_name + generator 
    exp_dir = "./exp/" + exp_name
    os.mkdir(exp_dir)
    log_file = exp_dir + '/exp.log'
    file = open(log_file, 'w')
    file.write("train log\n")
    file.close()
    config_file = exp_dir + '/config.yml'
    param_dir = exp_dir + "/param"
    os.mkdir(param_dir)
   
    return [exp_name, exp_dir, log_file, config_file, param_dir]


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
     
