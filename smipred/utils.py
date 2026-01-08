import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    # Sets the seed for Python, NumPy and PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sets the seed for CUDA operations on GPU to ensure reproducibility across multiple GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # This setting optimizes CUDA kernel selection for performance, which may reduce reproducibility across different runs
    torch.backends.cudnn.benchmark = True

    return
