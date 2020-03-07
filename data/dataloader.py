from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

class SudokuDataset(Dataset):
    def __init__(self, file, mono=True, cap_train=None):
        self.data = pd.read_csv(file, nrows=cap_train)
        self.mono = mono
        self.cap_train = cap_train
        
    def __len__(self):
        if self.cap_train is not None:
            return min(len(self.data), self.cap_train)
        return len(self.data)
       
    def __getitem__(self, i):
        row = self.data.iloc[i]
        if 'puzzle' in row:
            x = row['puzzle']
            y = row['solution']
        else:
            x = row['quizzes']
            y = row['solutions']
        if self.mono:
            x = self.to_mono_grid(x)
        else:
            x = self.to_stacked_grid(x)
        y = self.to_mono_grid(y).squeeze(0) -1
        return x, y
        
    @staticmethod    
    def to_mono_grid(x):
        res = np.zeros(81)
        for i in range(len(x)):
            res[i] = int(x[i])
        res = res.reshape((1,9,9))
        return res
    
    @staticmethod
    def to_stacked_grid(x):
        res = np.zeros((9,81))
        for i in range(len(x)):
            val = int(x[i])
            if val != 0:
                res[val-1][i] = 1
        res = res.reshape((9,9,9))
        return res
    
def get_loader(root, train=True, mono=True, batch_size=42, cap_train=None):
    if train:
        source = root + 'sudoku.csv'
    else:
        source = root + 'sudoku_test.csv'
        
    dataset = SudokuDataset(source, mono, cap_train)
    return DataLoader(dataset, shuffle=train, batch_size=batch_size)
        