import torch

from .isudokuenv import ISudokuEnv

class GridEnv(ISudokuEnv):
    '''
    Params:
        state: tensor (BSx1x9x9) representing current current sudoku board
        action: tensor (BSx81*9) representing the options to place a number in a cell
        goal: tensor (BSx9x9) representing the ground truth completed puzzle

    Returns:
        new_states: tensor (BSx1x9x9) representing new board
        rewards: tensor (BSx1) representing reward for each state/action/goal pair
        done: list of BS booleans representing if that puzzle is complete 
    '''
    @staticmethod
    def act(state, action, goal):
        new_states = []
        rewards = []
        done = []
        for i in range(len(state)):
            current = state[i, :, :, :]
            act = action[i, :]
            complete = goal[i, :]
            argmax = act.argmax().item()
            coord = argmax % 81
            val = (argmax // 81) + 1
            x = coord // 9
            y = coord % 9

            if current[0, x, y] != 0:
                rewards.append([-2])
                new_states.append(current)
                done.append(False)
                continue

            quadrant_start_x = x // 3
            quadrant_start_y = y // 3
            if val in current[0, x, :] or val in current[0, :, y] or \
                    val in current[0, quadrant_start_x:quadrant_start_x+3, quadrant_start_y:quadrant_start_y+3]:
                rewards.append([-1])
                new_state = current.clone()
                new_state[0, x, y] = complete[x, y] + 1
                new_states.append(new_state)
                d = False
                if 0 not in new_state:
                    d = True
                done.append(d)
                continue

            if complete[x, y] == val-1:
                new_state = current.clone()
                new_state[0, x, y] = val
                new_states.append(new_state)
                if 0 not in new_state:
                    rewards.append([10])
                    done.append(True)
                    continue
                if 0 not in new_state[0, x, :] or 0 not in new_state[0, :, y] or \
                        0 not in new_state[0, quadrant_start_x:quadrant_start_x+3, quadrant_start_y:quadrant_start_y+3]:
                    rewards.append([1])
                    done.append(False)
                    continue

            rewards.append([0])
            new_state = current.clone()
            new_state[0, x, y] = complete[x, y] + 1
            new_states.append(new_state)
            d = False
            if 0 not in new_state:
                d = True
            done.append(d)

        return torch.stack(new_states).squeeze(0), torch.tensor(rewards), done
