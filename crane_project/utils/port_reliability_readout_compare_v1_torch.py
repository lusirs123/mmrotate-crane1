"""Identical hidden initialization; sole treatment is the risk readout."""
from copy import deepcopy

import torch
from torch import nn

from crane_project.utils import port_reliability_redc_size_v1_torch as old

exported = old.exported
optimizer = old.optimizer
update = old.update


class ResidualRisk(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.network = deepcopy(temperature.network)
        # Zero output is the neutral residual. Temperature's neutral output is
        # inverse-softplus(.75). Both identities give exactly the same -z.
        nn.init.zeros_(self.network[4].bias)
        self.beta = nn.Parameter(torch.zeros_like(temperature.beta))

    def forward(self, normalized, raw_logit):
        return -raw_logit+self.network(normalized[:,1:]).squeeze(-1)+self.beta[0]


def make_models(normalizer, device):
    # Preserve even the RNG consumption of the old linear control. This makes
    # the temperature arm a reproducible replication, not a new initialization.
    temperature = old.make_models(normalizer, device)['redc']
    residual = ResidualRisk(temperature).to(device)
    if any(not torch.equal(v,residual.network.state_dict()[k])
           for k,v in temperature.network.state_dict().items() if not k.startswith('4.')):
        raise ValueError('Hidden initial weights differ between readouts')
    return dict(temperature=temperature,residual=residual)
