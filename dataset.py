from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import gc



class MyDataset(Dataset):
    def __init__(self, filepath):
        super(MyDataset, self).__init__()
        data = pd.read_csv(filepath)
        self.train_data = data.values[:,:-1]
        self.train_target = data.values[:,-1]
        self.length = data.shape[0]
        del data
        gc.collect()
    def __getitem__(self, index):
        
        return self.train_data[index,:], self.train_target[index]
        
    def __len__(self):
        return self.length