import torch
import logging

# Set device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_logger(logger_name, log_file, level=logging.INFO):
    """Creates and returns a logger that logs to both console and file."""
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    return logger


def configure_device():
    """Returns the current device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_random_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Ensure device and seeds are properly initialized
set_random_seeds()
