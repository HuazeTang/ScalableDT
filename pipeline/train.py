import torch
import torch.optim as optim
import torch.nn as nn  
from torch.nn.parallel import DistributedDataParallel as DDP  
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import datetime
import os

from model import SimpleModel  
from utils import setup, cleanup  

# Define the training function
def train(rank, world_size, epochs=5, accumulation_steps=2, checkpoint_path='./checkpoints'):
    setup(rank, world_size)

    # Logging setup
    log_dir = f'./logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(log_dir) if rank == 0 else None

    # Data augmentation and CIFAR-10 dataset loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)

    # Setup model, loss function, optimizer, and learning rate scheduler
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = GradScaler()  # Used for mixed precision training

    # Checkpoint save path
    os.makedirs(checkpoint_path, exist_ok=True)

    # Training loop
    ddp_model.train()
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)

            # Gradient accumulation & mixed precision training
            optimizer.zero_grad()
            with autocast():
                output = ddp_model(data)
                loss = criterion(output, target) / accumulation_steps  # Reduce gradient by accumulation steps

            scaler.scale(loss).backward()
            
            # Update parameters after a certain number of accumulation steps
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()

        scheduler.step()  # Update learning rate

        # Checkpoint management
        if rank == 0 and (epoch + 1) % 2 == 0:  # Save every 2 epochs
            checkpoint_file = f'{checkpoint_path}/model_epoch_{epoch+1}.pth'
            torch.save(ddp_model.state_dict(), checkpoint_file)
            print(f'Checkpoint saved to {checkpoint_file}')

        # Logging
        if writer and rank == 0:
            writer.add_scalar('Loss/train', loss.item(), epoch)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
            print(f'Epoch {epoch+1}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}')

    cleanup()
    if writer:
        writer.close()
