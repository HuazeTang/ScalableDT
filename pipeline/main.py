import torch
from train import train
from utils import setup, cleanup

if __name__ == "__main__":
    world_size = 2  
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
