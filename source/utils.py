# check if gpu is available
import torch

def check_gpu():
    return torch.cuda.is_available()
