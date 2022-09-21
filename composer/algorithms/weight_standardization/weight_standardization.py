# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn
from torch.fx import symbolic_trace

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

__all__ = ['apply_weight_standardization', 'WeightStandardization']


def _standardize_weights(W: torch.Tensor):
    reduce_dims = list(range(1, W.dim()))
    W_var, W_mean = torch.var_mean(W, dim=reduce_dims, keepdim=True, unbiased=False)
    return (W - W_mean) / (torch.sqrt(W_var + 1e-5))


class WeightStandardizer(nn.Module):

    def forward(self, W):
        return _standardize_weights(W)


def batch_to_layer_norm(module: torch.nn.BatchNorm2d, module_index: int):
    layer_norm = torch.nn.LayerNorm(module.num_features, eps=module.eps, elementwise_affine=module.affine)

    if module.affine:
        with torch.no_grad():
            layer_norm.weight.copy_(module.weight)
            layer_norm.bias.copy_(module.bias)

    return layer_norm


def apply_weight_standardization(model: torch.nn.Module, n_last_layers_ignore: bool = False, optimizers=None):
    count = 0
    model_trace = symbolic_trace(model)
    for module in model_trace.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and module.groups == 1:
            parametrize.register_parametrization(module, 'weight', WeightStandardizer())
            count += 1

    target_ws_layers = count - n_last_layers_ignore

    for module in list(model_trace.modules())[::-1]:
        if target_ws_layers == count:
            break
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and module.groups == 1:
            parametrize.remove_parametrizations(module, 'weight', leave_parametrized=False)
            count -= 1

    transforms = {torch.nn.BatchNorm2d: batch_to_layer_norm}
    module_surgery.replace_module_classes(model, optimizers=optimizers, policies=transforms)
    gn_count = module_surgery.count_module_instances(model, torch.nn.GroupNorm)
    return count, gn_count


class WeightStandardization(Algorithm):
    """Weight standardization.

    Weight standardization.

    Args:
        n_last_layers_ignore (int): ignores the laste layer. Default: ``False``.
    """

    # TODO: Maybe make this ignore last n layers in case there are multiple prediction heads? Would this work?
    def __init__(self, n_last_layers_ignore: int = 0):
        self.n_last_layers_ignore = n_last_layers_ignore

    def match(self, event: Event, state: State):
        return (event == Event.INIT)

    def apply(self, event: Event, state: State, logger: Logger):
        count, gn_count = apply_weight_standardization(state.model,
                                                       n_last_layers_ignore=self.n_last_layers_ignore,
                                                       optimizers=state.optimizers)
        logger.data_fit({'WeightStandardization/num_weights_standardized': count})
        logger.data_fit({'WeightStandardization/num_layer_norms': gn_count})
