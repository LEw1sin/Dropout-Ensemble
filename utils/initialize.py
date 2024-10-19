import torch
import torch.distributed as dist
import os
import logging

def init_distributed_mode(args):
    args.distributed = True
    # Specify the GPU to be used by the current process
    torch.cuda.set_device(args.gpu)
    # Communication backend; NVIDIA GPUs recommend using NCCL
    args.dist_backend = 'nccl'
    args.dist_url = 'tcp://127.0.0.1:23456'  # Please ensure dist_url is set
    logging.info('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url))
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    # Wait until all GPUs reach this point before proceeding
    dist.barrier()
    # If using multiple machines with multiple GPUs, WORLD_SIZE represents the number of machines, RANK corresponds to each machine
    # If using a single machine with multiple GPUs, WORLD_SIZE represents the number of GPUs, RANK and LOCAL_RANK correspond to each GPU
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        logging.info('Not using distributed mode')
        args.distributed = False
        return

def get_world_size():
    if not dist.is_initialized():
        raise RuntimeError("Distributed package is not initialized.")
    return dist.get_world_size()

def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # Single GPU scenario
        return value

    with torch.no_grad():
        dist.all_reduce(value)   # Sum the values across different devices
        if average:  # If averaging is needed, compute the mean loss across multiple GPUs
            value /= world_size

        return value

def is_main_process():
    return dist.get_rank() == 0

def setup_logging(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
