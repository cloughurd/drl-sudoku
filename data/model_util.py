import numpy as np
import torch

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

def stacked_to_mono(puzzles):
    zeros_mask = torch.all(puzzles == 0, dim=1).cuda()
    maxes = puzzles.argmax(dim=1).cuda()
    return torch.where(zeros_mask, torch.zeros(zeros_mask.size()).cuda(), maxes.float().cuda() + 1).cuda()