import pandas as pd
import numpy as np

from .isudokuenv import ISudokuEnv

class GridEnv(ISudokuEnv):
    def __init__(self, file, max_len=None, weight=False):
        self.data = pd.read_csv(file, nrows=max_len)
        if weight:
            self.data['weight'] = self.data.puzzle.apply(lambda p: 81-p.count('0'))
        else:
            self.data['weight'] = 1
        
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
        row = self.data.sample(weights='weight').iloc[0]
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
    def act(state, action, goal):
        coord = action % 81
        val = (action // 81)
        x = coord // 9
        y = coord % 9

        if 1 in state[:, x, y]:
            return state, -3, False

        quadrant_start_x = x // 3
        quadrant_start_y = y // 3
        if 1 in state[val, x, :] or 1 in state[val, :, y] or \
                1 in state[val, quadrant_start_x:quadrant_start_x+3, quadrant_start_y:quadrant_start_y+3]:
            r = -1
            new_state = state.copy()
            new_state[int(goal[x,y]), x, y] = 1
            d = False
            if np.sum(new_state) == 81:
                return new_state, -2, True
            else:
                return new_state, -2, False

        if goal[x, y] == val:
            new_state = state.copy()
            new_state[val, x, y] = 1
            if np.sum(new_state) == 81:
                return new_state, 10, True
            if np.sum(new_state[:, x, :]) == 9 or np.sum(new_state[:, :, y]) == 9 or \
                    np.sum(new_state[:, quadrant_start_x:quadrant_start_x+3, quadrant_start_y:quadrant_start_y+3]) == 9:
                return new_state, 5, False
            else:
                return new_state, 1, False

        new_state = state.copy()
        new_state[int(goal[x,y]), x, y] = 1
        if np.sum(new_state) == 81:
            return new_state, 0, True
        else:
            return new_state, 0, False
