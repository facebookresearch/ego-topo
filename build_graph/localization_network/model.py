import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as tmodels

class SiameseR18_5MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.trunk = tmodels.resnet18(pretrained=True)
        self.trunk.fc = nn.Sequential()
        self.compare = nn.Sequential(
                            nn.Linear(512*2, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 2))

    def forward(self, batch, softmax=False):

        featA, featB = self.trunk(batch['imgA']), self.trunk(batch['imgB']) 
        featAB = torch.cat([featA, featB], dim=1)
        sim_pred = self.compare(featAB)

        loss_dict = {}
        if 'label' in batch:
            loss = F.cross_entropy(sim_pred, batch['label'], reduction='none')
            loss_dict.update({'sim': loss})

        if softmax:
            sim_pred = F.softmax(sim_pred, 1)[:, 1]
        
        return sim_pred, loss_dict


class R18_5MLP(SiameseR18_5MLP):
    def __init__(self):
        super().__init__()
        self.compare = nn.Sequential(
                            nn.Linear(512*2, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 2))
        
        self.trunk = nn.Identity()

