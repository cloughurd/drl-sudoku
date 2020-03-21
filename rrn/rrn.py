import torch
import torch.nn as nn
import torch.nn.functional as F
from message_passing import message_passing

class SudokuRecurrentRelationalNet(nn.Module):
    def __init__(self):
        super(SudokuRecurrentRelationalNet, self).__init__()
        self.global_step = 0
        self.num_steps = 32
        self.mlp_hidden = 96
        self.lstm_hidden = self.mlp_hidden
        self.embed_size = 16
        self.embed = nn.Embedding(10, self.embed_size)
        self.mlp_depth = 3
        self.pre = self.get_mlp(self.mlp_hidden, self.mlp_depth)
        self.post = self.get_mlp(self.mlp_hidden, self.mlp_depth)
        self.update = nn.LSTM(self.mlp_hidden, self.lstm_hidden)
        self.message_fn = self.get_mlp(self.mlp_hidden, self.mlp_depth)
        self.output = nn.Linear(96, 10)

    def get_mlp(self, n_hidden, depth):
        return nn.Sequential(
            nn.Linear(10, n_hidden),
            *[item
                for pair in 
                [( nn.ReLU(), nn.Linear(n_hidden, n_hidden)) for _ in range(depth - 1)]
                for item in pair]
        )

    def forward(self, puzzle, nodes, edges, edge_features, state, first=False):
        """
            The forward pass on a bunch of puzzles - 
            \n!!! Is one message passing and update instance.
        """
        # on the first pass through I think we need to embed.
        # We could theoretically do this before hand, but the Embedding
        # Is trainable, so having it inside the model is best.
        if first:
            x = self.embed(nodes)
        else:
            x = nodes

        x = message_passing(x, edges, edge_features, self.message_fn)
        x = self.post(torch.cat( [x, puzzle], dim=1))
        x, state = self.update(x, state)
        out = self.output(x).view((-1, 81, 10))
        return x, state, out

