import pandas as pd
import numpy as np

from .isudokuenv import ISudokuEnv

class SisypheanGrid(ISudokuEnv):
    def __init__(self, file, max_len=None):
        self.data = pd.read_csv(file, nrows=max_len)
        
    @staticmethod    
    def to_mono_grid(x):
        res = np.zeros(81)
        for i in range(len(x)):
            res[i] = int(x[i])
        res = res.reshape((9,9))
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

    def reset(self):
        row = self.data.sample().iloc[0]
        if 'puzzle' in row:
            x = row['puzzle']
            y = row['solution']
        else:
            x = row['quizzes']
            y = row['solutions']
        
        x = self.to_stacked_grid(x)
        y = self.to_mono_grid(y) -1
        return x, y

    '''
    Params:
        state: np array (9x9x9) representing current current sudoku board
        action: int (from 0 to 81*9) representing the options to place a number in a cell
        goal: np array (9x9) representing the ground truth completed puzzle

    Returns:
        new_state: np array (9x9x9) representing new board
        reward: int representing reward for state/action/goal pair
        done: boolean representing if that puzzle is complete 
    '''
    @staticmethod
    def act(state, action, goal, reward_magnitude=1):
        coord = action % 81
        val = (action // 81)
        x = coord // 9
        y = coord % 9

        if 1 in state[:, x, y]:
            return state, -reward_magnitude, False

        if goal[x, y] == val:
            new_state = state.copy()
            new_state[val, x, y] = 1
            reward = reward_magnitude
        else:
            #If it didn't get it right, don't change anything.
            new_state = state
            reward = -reward_magnitude

        return new_state, reward, np.sum(new_state) == 81
