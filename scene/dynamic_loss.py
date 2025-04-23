import torch
import torch.nn as nn


class DynamicWeightedLoss(nn.Module):
    def __init__(
        self,
        keys=["rgb", "semantics"],):
        super(DynamicWeightedLoss, self).__init__()
        self.params = nn.ParameterDict(
            {k: nn.Parameter(torch.ones(1, requires_grad=True)) for k in keys}
        )
        self.params["rgb"] = self.params["rgb"] * 2.0
        self.params["semantics"] = self.params["semantics"] * 2.0

    def forward(self, x):
        #loss_sum = 0
        #keys = ["rgb", "semantics"]
        #for k in x.keys():
            #print(self.params[k])
        #print((self.params["rgb"] + self.params["semantics"]))
        #self.params["rgb"] = (2 * self.params["rgb"] / (self.params["rgb"] + self.params["semantics"]))
        #self.params["semantics"] = (2 * self.params["semantics"] / (self.params["rgb"] + self.params["semantics"]))
        #print(self.params["rgb"])
        #print(self.params["semantics"])
        loss_sum = torch.exp(-self.params["rgb"]) * x["rgb"] + self.params["rgb"]  \
                   + torch.exp(-self.params["semantics"]) * x["semantics"] + self.params["semantics"]
        return loss_sum