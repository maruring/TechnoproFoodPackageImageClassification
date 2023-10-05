import torch
import torch.nn as nn

class MyEnsembleModel(nn.Module):
    def __init__(self, modelA, modelB, modelC, modelD, modelE, input):
        super(MyEnsembleModel, self).__init__()

        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.modelE = modelE

        self.add_fc = nn.Linear(input, 2)

    def forward(self, x):
        outA = self.modelA(x)
        outB = self.modelB(x)
        outC = self.modelC(x)
        outD = self.modelD(x)
        outE = self.modelE(x)

        out = outA + outB + outC + outD + outE

        x = self.add_fc(out)
        return torch.softmax(x, dim=1)