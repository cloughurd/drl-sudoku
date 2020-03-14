import random
import torch

def get_action_dqn(network, state, epsilon, epsilon_decay, device='cuda'):
  """Select action according to e-greedy policy and decay epsilon

    Args:
        network (QNetwork): Q-Network
        state (np-array): current state, size (state_size)
        epsilon (float): probability of choosing a random action
        epsilon_decay (float): amount by which to decay epsilon

    Returns:
        action (int): chosen action [0, action_size)
        epsilon (float): decayed epsilon
  """
  act_randomly = random.random()
  if act_randomly < epsilon:
    action = random.randrange(0, network.output_size)
  else:
    pred = network(torch.from_numpy(state).to(device=device).float().unsqueeze(0))
    action = torch.argmax(pred.reshape(1, 81*9))
    action = action.item()
  epsilon *= epsilon_decay
  return action, epsilon


def prepare_batch(memory, batch_size, device='cuda'):
  """Randomly sample batch from memory
     Prepare cuda tensors

    Args:
        memory (list): state, action, next_state, reward, done tuples
        batch_size (int): amount of memory to sample into a batch

    Returns:
        state (tensor): float cuda tensor of size (batch_size x state_size)
        action (tensor): long tensor of size (batch_size)
        next_state (tensor): float cuda tensor of size (batch_size x state_size)
        reward (tensor): float cuda tensor of size (batch_size)
        done (tensor): float cuda tensor of size (batch_size)
  """
  batch = random.sample(memory, batch_size)
  state, action, next_state, reward, done = zip(*batch)
  state = torch.tensor(state).to(device=device).float()
  action = torch.tensor(action).long()
  next_state = torch.tensor(next_state).to(device=device).float()
  reward = torch.tensor(reward).to(device=device).float()
  done = torch.tensor(done).to(device=device).float()
  return state, action, next_state, reward, done
  
  
def learn_dqn(batch, optim, q_network, target_network, gamma, global_step, target_update, device='cuda'):
  """Update Q-Network according to DQN Loss function
     Update Target Network every target_update global steps

    Args:
        batch (tuple): tuple of state, action, next_state, reward, and done tensors
        optim (Adam): Q-Network optimizer
        q_network (QNetwork): Q-Network
        target_network (QNetwork): Target Q-Network
        gamma (float): discount factor
        global_step (int): total steps taken in environment
        target_update (int): frequency of target network update
  """
  optim.zero_grad()
  state, action, next_state, reward, done = batch

  q_output = q_network(state).reshape(-1, 81*9)
  q_output = torch.gather(q_output, 1, action.reshape(-1,1).to(device=device)).squeeze(1)

  target_output = target_network(next_state).reshape(-1, 81*9)
  target_output, _ = torch.max(target_output, axis=1)

  loss = reward + (gamma * target_output)
  loss = loss * (1 - done)
  loss = q_output - loss
  loss = loss ** 2
  loss = torch.mean(loss)
  loss.backward()
  optim.step()

  if global_step % target_update == 0:
    # print('target network updated')
    target_network.load_state_dict(q_network.state_dict())

  return loss.item()