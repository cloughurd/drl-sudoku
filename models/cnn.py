import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResBlock, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.Dropout(),
        )

        if in_c != out_c:
            self.skip = nn.Conv2d(in_c, out_c, kernel_size=1)
        else:
            self.skip = nn.Identity()
        self.final = nn.ReLU()

    def forward(self, x):
        res = self.net(x)
        y = self.skip(x) + res
        return self.final(y)
    
class Flattener(nn.Module):
    def __init__(self):
        super(Flattener, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        pooled = self.pool(x)
        return x.squeeze(2).squeeze(2)

class BasicNet(nn.Module):
    def __init__(self, in_c, num_blocks=5):
        super(BasicNet, self).__init__()
        c = 16
        layers = [ResBlock(in_c, 16)]
        for i in range(1, num_blocks):
            next_c = c
            if i % 2 == 0:
                next_c *= 2
            layers.append(ResBlock(c, next_c))
            c = next_c
        layers.append(nn.Flatten())
        layers.append(nn.Linear(81*c, 81*9))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        y = self.net(x).reshape((-1, 81, 9))
        return y
