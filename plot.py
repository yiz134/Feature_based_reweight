import torch
import matplotlib.pyplot as plt
from utils import *
w_list = torch.load("no_recompute.pth")

plot_weights_hist(w_list[-1].cpu(), train_clean_idx, train_noise_idx, val_idx)