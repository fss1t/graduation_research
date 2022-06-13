# coding:utf-8

import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def f_loss_adv(logits, target):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape(-1)
    targets = torch.full_like(logits, fill_value=target)
    logits = logits.clamp(min=-10, max=10)  # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def f_reg_r1(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def f_f0_mean(f0, vuv):
    f0_sum = torch.sum(f0 * vuv, -1, True)
    vuv_sum = torch.sum(vuv, -1, True)
    vuv_sum += vuv_sum == 0 # prevent division by 0

    f0_mean = f0_sum / vuv_sum
    return f0_mean


def f_loss_f0(f0_x, vuv_x, f0_y, vuv_y):
    f0_x_mean = f_f0_mean(f0_x, vuv_x)
    f0_y_mean = f_f0_mean(f0_y, vuv_y)

    vuv_and = vuv_x * vuv_y
    loss = F.smooth_l1_loss((f0_x - f0_x_mean) * vuv_and, (f0_y - f0_y_mean) * vuv_and)
    return loss
