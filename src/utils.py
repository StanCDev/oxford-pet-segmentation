import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from torch.utils.data import TensorDataset, DataLoader


# Generaly utilies
##################
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



# Metrics
#########
def IoU(y, y_pred):
    return 0