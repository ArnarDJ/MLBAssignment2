import torch

DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"

BATCH_SIZE = 128
NUM_WORKERS = 2
LEARNING_RATE = 1e-3
LATENT_DIM = 64
EPOCHS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")