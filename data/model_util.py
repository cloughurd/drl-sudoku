import numpy as np
import torch

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

def stacked_to_mono(puzzles, cuda=True):
    zeros_mask = torch.all(puzzles == 0, dim=1)
    maxes = puzzles.argmax(dim=1).float() + 1
    zeros = torch.zeros(zeros_mask.size())
    
    if cuda:
        zeros_mask = zeros_mask.cuda(async=False)
        maxes = maxes.cuda(async=False)
        zeros = zeros.cuda(async=False)

    result = torch.where(zeros_mask, zeros, maxes)

    if cuda:
        result = result.cuda(async=False)

    return result
