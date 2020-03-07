import numpy as np
import torch

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

def stacked_to_mono(puzzles):
    zeros_mask = torch.all(puzzles == 0, dim=0)
    maxes = puzzles.argmax(dim=0)
    return torch.where(zeros_mask, torch.zeros(zeros_mask.size()), maxes.float() + 1)