import csv
import os
import urllib.request
import zipfile


from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
 
def sudoku_edges():
    def cross(a):
        return [[i, j] for i in a.flatten() for j in a.flatten() if not i == j]

    idx = np.arange(81).reshape(9, 9)
    rows, columns, squares = [], [], []
    for i in range(9):
        rows += cross(idx[i, :])
        columns += cross(idx[:, i])
    for i in range(3):
        for j in range(3):
            squares += cross(idx[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3])
    return list(set(rows + columns + squares))

class sudoku:
    url = "https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1"  # See generate_hard.py on how this dataset was generated
    zip_fname = "/tmp/sudoku-hard.zip"
    dest_dir = '/tmp/sudoku-hard/'

    def __init__(self):
        if not os.path.exists(self.dest_dir):
            print("Downloading data...")

            urllib.request.urlretrieve(self.url, self.zip_fname)
            with zipfile.ZipFile(self.zip_fname) as f:
                f.extractall('/tmp/')

        def read_csv(fname):
            print("Reading %s..." % fname)
            with open(self.dest_dir + fname) as f:
                reader = csv.reader(f, delimiter=',')
                return [(q, a) for q, a in reader]

        self.train = read_csv('train.csv')
        self.valid = read_csv('valid.csv')
        self.test = read_csv('test.csv')

class SudokuGraphDataset(Dataset):
    def __init__(self, file, mono=True, cap_train=None, sort=False):
        if sort:
            csv = pd.read_csv(file)
            empty_counts = csv['puzzle'].apply(lambda p: p.count('0'))
            csv['num_empty'] = empty_counts
            csv.sort_values(by="num_empty", inplace=True)

            if cap_train is not None:
                csv = csv.head(cap_train)

            self.data = csv
        else:
            self.data = pd.read_csv(file, nrows=cap_train)
        self.mono = mono
        self.cap_train = cap_train
        self.edges = sudoku_edges()
        
    def __len__(self):
        if self.cap_train is not None:
            return min(len(self.data), self.cap_train)
        return len(self.data)
    
    def _parse(self, x):
        return list(map(int, list(x)))  

    def __getitem__(self, i):
        row = self.data.iloc[i]
        if 'puzzle' in row:
            x = row['puzzle']
            y = row['solution']
        else:
            x = row['quizzes']
            y = row['solutions']

        x, y = self._parse(x), self._parse(y)
        x, y = np.array(x, np.int32), np.array(y, np.int32)

        return (x, np.array(self.edges)), y
        
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