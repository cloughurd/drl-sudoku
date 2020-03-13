from .isudokuenv import ISudokuEnv

class CoordinateEnv(ISudokuEnv):
    '''
    Params:
        state: tuple of tensors ((BSx2), (BSx1x9x9)) representing current position and current sudoku board
        action: tensor (BSx13) representing the options to move around the board or place a number
        goal: tensor (BSx9x9) representing the ground truth completed puzzle

    Returns:
        new_state: tuple of tensors ((BSx2), (BSx1x9x9)) represent new position and board
        reward: tensor (BSx1) representing reward for each state/action/goal pair
        done: boolean representing whether the game is complete
    '''
    @staticmethod
    def act(state, action, goal):
        for i in range(len(state)):
            pass