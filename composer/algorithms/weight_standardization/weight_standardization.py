# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

__all__ = ['apply_weight_standardization', 'WeightStandardization']


def _standardize_weights(W: torch.Tensor):
    reduce_dims = list(range(1, W.dim()))
    W_var, W_mean = torch.var_mean(W, dim=reduce_dims, keepdim=True, unbiased=False)
    return (W - W_mean) / (torch.sqrt(W_var + 1e-10))


class WeightStandardizer(nn.Module):

    def forward(self, W):
        return _standardize_weights(W)


def batch_to_group_norm(module: torch.nn.BatchNorm2d, module_index: int):
    group_norm = torch.nn.GroupNorm(num_groups=8,
                                    num_channels=module.num_features,
                                    eps=module.eps,
                                    affine=module.affine)

    if module.affine:
        with torch.no_grad():
            group_norm.weight.copy_(module.weight)
            group_norm.bias.copy_(module.bias)

    return group_norm


def apply_weight_standardization(model: torch.nn.Module):
    ws_count = 0
    for module in model.modules():
        if (isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d)):
            parametrize.register_parametrization(module, 'weight', WeightStandardizer())
            ws_count += 1
    transforms = {torch.nn.BatchNorm2d: batch_to_group_norm}
    module_surgery.replace_module_classes(model, policies=transforms)
    gn_count = module_surgery.count_module_instances(model, torch.nn.GroupNorm)
    return ws_count, gn_count


class WeightStandardization(Algorithm):

    def __init__(self):
        pass

    def match(self, event: Event, state: State):
        return (event == Event.INIT)

    def apply(self, event: Event, state: State, logger: Logger):
        ws_count, gn_count = apply_weight_standardization(state.model)
        logger.log_hyperparameters({'WeightStandardization/num_weights_standardized': ws_count})
        logger.log_hyperparameters({'WeightStandardization/num_group_norms': gn_count})
