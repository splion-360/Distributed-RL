
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import os

def distributed_training(rank, world_size):
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Print the rank and the process group backend
    print(f"Rank {rank} initialized process group with backend: {dist.get_backend()}")

    # Simulate training by creating a simple tensor
    tensor = torch.tensor([rank] * 10)
    # All-reduce the tensor across all processes
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} has tensor {tensor}")

    # Cleanup
    dist.destroy_process_group()

def main():
    # Number of processes (should match the number of nodes you want to use)
    world_size = 4

    # Spawn processes for each node
    processes = []
    for rank in range(world_size):
        p = Process(target=distributed_training, args=(rank, world_size))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
