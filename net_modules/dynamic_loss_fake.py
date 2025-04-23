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

    def forward(self, x):
        print(x)
        print("-----------")
        loss_sum = 0
        keys = ["rgb", "semantics"]
        #for k in x.keys():
            #print(self.params[k])
        print((self.params["rgb"] + self.params["semantics"]))
        self.params["rgb"] = (2 * self.params["rgb"] / (self.params["rgb"] + self.params["semantics"]))
        self.params["semantics"] = (2 * self.params["semantics"] / (self.params["rgb"] + self.params["semantics"]))
        print(self.params["rgb"])
        print(self.params["semantics"])
        loss_sum = (0.5 / (self.params["rgb"] ** 2) * x["rgb"]) + torch.log(1 + self.params["rgb"] ** 2) \
                   + (0.5 / (self.params["semantics"] ** 2) * x["semantics"]) + torch.log(1 + self.params["semantics"] ** 2)
        return loss_sum