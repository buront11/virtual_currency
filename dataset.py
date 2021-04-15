import torch
import torch.nn
from torch.utils.data import Dataset
import torchvision

class VirtualDataset(Dataset):
    def __init__(self):
        super(VirtualDataset,self).__init__()
        
