from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

class SudokuDataset(Dataset):
    def __init__(self, file, mono=True):
        self.data = pd.read_csv(file)
        self.mono = mono
        
    def __len__(self):
        return len(self.data)
       
    def __getitem__(self, i):
        row = self.data.iloc[i]
        x = row['puzzle']
        y = row['solution']
        if self.mono:
            x = self.to_mono_grid(x)
        else:
            x = self.to_stacked_grid(x)
        y = self.to_mono_grid(y)
        return x, y
        
    @staticmethod    
    def to_mono_grid(x):
        print(x)
        res = np.zeros(81)
        for i in range(len(x)):
            res[i] = int(x[i])
        res = res.reshape((9,9))
        print(res)
        return res
    
    @staticmethod
    def to_stacked_grid(x):
        print(x)
        res = np.zeros((9,81))
        for i in range(len(x)):
            val = int(x[i])
            if val != 0:
                res[val-1][i] = 1
        res = res.reshape((9,9,9))
        print(res)
        return res
    
def get_loader(train=True, mono=True, batch_size=42):
    if train:
        source = 'sudoku.csv'
    else:
        source = 'sudoku_test.csv'
        
    dataset = SudokuDataset(source, mono)
    return DataLoader(dataset, shuffle=train, batch_size=batch_size)
        