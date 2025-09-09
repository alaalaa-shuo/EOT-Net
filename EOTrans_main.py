import os
import random
import torch
import numpy as np
import Trans_mod


def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Device Configuration
data = 'samson'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
get_random_seed(2)
print("\nSelected device:", device, end="\n\n")

tmod = Trans_mod.Train_test(dataset=data, device=device, skip_train=False, save=True, data_print=True)
tmod.run(smry=False)
