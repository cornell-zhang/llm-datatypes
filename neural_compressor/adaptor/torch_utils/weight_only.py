#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 MIT HAN Lab
# This source code is licensed under the MIT license
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from copy import deepcopy
from typing import OrderedDict

from ...utils import logger
from ...utils.utility import LazyImport
from .util import set_module

tqdm = LazyImport("tqdm")
torch = LazyImport("torch")

NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, 
       -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]

NF3 = [-1.0, -0.5350227084855379, -0.24693143181157517, 0.0, 0.18333748033548752, 0.3819939543209139, 0.6229857417143423, 1.0]

NF3_NZ = [-1.0, -0.5840457615163138, -0.3225133196607414, -0.10387883073852357, 0.10387883073852357, 0.3225133196607414, 0.5840457615163137, 1.0]

SF4_3 = [-1.0, -0.575725, -0.403755, -0.29162, -0.204868, -0.131054, -0.064036,
         0.0, 0.055934, 0.113835, 0.176078, 0.246078, 0.329642, 0.438727, 0.60598, 1.0
]
SF3_3 = [-1.0, -0.3622475096628733, -0.15466881612248634, 
         0.0, 0.11383526480516333, 0.24607773854836362, 0.43872699492795214, 1.0
]
SF3_3_NZ = [-1.0, -0.5015162581807956, -0.2628875575162947, -0.08299175634976376, 0.08299175634976376, 0.2628875575162947, 0.5015162581807955, 1.0]

SF4_4 = [-1.0, -0.608729, -0.435845, -0.318393, -0.225229, -0.144689, -0.070863,
         0.0, 0.061908, 0.125773, 0.193936, 0.269711, 0.358617, 0.471724, 0.638231, 1.0
]
SF3_4 = [ -1.0, -0.39278935395019254, -0.1705619364387109, 
         0.0, 0.1257725217978133, 0.26971116287994534, 0.4717241438815071, 1.0
]
SF3_4_NZ = [-1.0, -0.5238993613586714, -0.27862916172493263, -0.08844753653560009, 0.08844753653560009, 0.27862916172493263, 0.5238993613586711, 1.0]

SF4_5 = [-1.0, -0.627781, -0.454736, -0.334331, -0.237434, -0.152899, -0.074982, 0.0,
         0.065513, 0.132965, 0.204661, 0.283835, 0.375805, 0.491076, 0.656781, 1.0
]
SF3_5 = [-1.0, -0.41084770299806544, -0.1801188154332576, 
         0.0, 0.1329647268004446, 0.2838347036415534, 0.49107557131011725, 1.0
]
SF3_5_NZ = [-1.0, -0.5368359642710748, -0.2878651877735525, -0.09166749796557234, 0.09166749796557234, 0.2878651877735525, 0.5368359642710749, 1.0]

SF4_6 = [-1.0, -0.640101, -0.467104, -0.344838, -0.245517, -0.15835, -0.077722,
         0.0, 0.067911, 0.137742, 0.211771, 0.293169, 0.387111, 0.503714, 0.66875, 1.0
]
SF3_6 = [-1.0, -0.4227024811287854, -0.18645957091951085, 
          0.0, 0.1377424733823289, 0.29316883983286, 0.5037142212421039, 1.0
]
SF3_6_NZ = [-1.0, -0.5452285431525636, -0.2939135955820373, -0.09378393351237442, 0.09378393351237442, 0.2939135955820373, 0.5452285431525635, 1.0]

SF4_7 = [-1.0, -0.648695668605137, -0.4758049192191136, -0.35226650560941064, -0.25124728719118344, -0.16222176644114206, -0.07967003587244491,
         0.0, 0.06961628484673037, 0.1411376777420972, 0.21681690757912492, 0.2997785136806453, 0.39509153065438735, 0.5125912352606964, 0.6770872505232854, 1.0]
FP4_BNB = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -0.0625, 0, 0.0625, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_E2M1 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.0625, 0, 0.0625, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_E1M2 = [-3.5, -3.0, -2.5, -2.0, -1.75, -1.5, -1.25, 0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5]
FP4_BASIC = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_NO_SUB = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.75, 0, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_RANGE = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
FP4_PREC1 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
FP4_PREC2 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
FP4_LOG = [-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
FP3_LOG = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
EN_25 = [-0.787862, -0.609476, -0.459471, -0.333333, -0.227264, -0.138071, -0.063069, 0.0,
         0.063069, 0.138071, 0.227264, 0.333333, 0.459471, 0.609476, 0.787862, 1.0]
APOT4 = [-1.0,  -0.8, -0.6, -0.4, -0.3, -0.2, -0.1,  0.,   0.1,  0.2,  0.3,  0.4,  0.6,  0.8, 1.0 ]
APOT4_SP = [-1.0,  -0.8, -0.6, -0.4, -0.3, -0.2, -0.1,  0.,   0.1,  0.2,  0.3,  0.4, 0.5, 0.6,  0.8, 1.0 ]
MOKEY = [-16, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 16]
DENORM4 = [-7, -6, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5, 6]
# the order is the same as float list, bit value range is [-7, 7]
# 1111 = -1, 1110 = -2, 1101= -3, ...

NF4_BIT = [7, 1, 2, 3, 4, 5, 6, 0, -8, -7, -6, -5, -4, -3, -2, -1]
NF3_BIT = [3, 1, 2, 0, -4, -3, -2, -1]
FP4_BNB_BIT = [-5, -6, -3, -4, -1, -2, -7, 0, 1, 6, 7, 4, 5, 2, 3]
FP4_E2M1_BIT = [-1, -2, -3, -4, -5, -6, -7, 0, 1, 2, 3, 4, 5, 6, 7]
FP3_BIT = [-1, -2, -3, -4, 0, 1, 2, 3]
FP4_SUPER_BIT = [-8, -1, -2, -3, -4, -5, -6, -7, 0, 1, 2, 3, 4, 5, 6, 7]

FLOAT_MAPPING = {"nf4": NF4, "nf3": NF3, "nf3_nz": NF3_NZ, "sf4_3": SF4_3, "sf3_3": SF3_3, "sf3_3_nz": SF3_3_NZ, "sf4_4": SF4_4,  "sf3_4": SF3_4,  "sf3_4_nz": SF3_4_NZ,
                 "sf4_5": SF4_5, "sf3_5": SF3_5, "sf3_5_nz": SF3_5_NZ, "sf4_6": SF4_6, "sf3_6": SF3_6, "sf3_6_nz": SF3_6_NZ, "sf4_7": SF4_7, "fp4": FP4_BNB, "fp4_e2m1_bnb": FP4_BNB,
                 "fp4_e1m2": FP4_E1M2, "fp4_e2m1": FP4_E2M1, "fp4_basic": FP4_BASIC, "fp4_no_sub": FP4_NO_SUB,
                 "fp4_range": FP4_RANGE, "fp4_prec1": FP4_PREC1, "fp4_prec2": FP4_PREC2,
                 "fp4_log": FP4_LOG, "fp3_log": FP3_LOG, "en_25": EN_25,
                 "apot4": APOT4, "apot4-sp": APOT4_SP, "mokey": MOKEY, "denorm4": DENORM4}
INT_MAPPING = {"nf4": NF4_BIT, "nf3": NF3_BIT, "nf3_nz": NF3_BIT, "sf4_3": NF4_BIT, "sf3_3": NF3_BIT, "sf3_3_nz": NF3_BIT, "sf4_4": NF4_BIT, "sf3_4": NF3_BIT, "sf3_4_nz": NF3_BIT,
               "sf4_5": NF4_BIT, "sf3_5": NF3_BIT, "sf3_5_nz": NF3_BIT, "sf4_6": NF4_BIT, "sf3_6": NF3_BIT, "sf3_6_nz": NF3_BIT, "sf4_7": NF4_BIT, "fp4": FP4_BNB_BIT, "fp4_e2m1_bnb": FP4_BNB_BIT,
               "fp4_e1m2": FP4_E2M1_BIT, "fp4_e2m1": FP4_E2M1_BIT, "fp4_basic": FP4_E2M1_BIT, "fp4_no_sub": FP4_E2M1_BIT,
               "fp4_range": FP4_SUPER_BIT, "fp4_prec1": FP4_SUPER_BIT, "fp4_prec2": FP4_SUPER_BIT,
               "fp4_log": FP4_E2M1_BIT, "fp3_log": FP3_BIT, "en_25": NF4_BIT,
               "apot4": FP4_E2M1_BIT, "apot4-sp": FP4_SUPER_BIT, "mokey": FP4_SUPER_BIT,
               "denorm4": FP4_SUPER_BIT}

def quantize_4bit(tensor, quantile=1.0, data_type="nf4", return_int=False, scale=None):
    """Quantize tensor to NF4/FP4 data type.

    Args:
        tensor: input tensor
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        data_type (str, optional): data type. Defaults to 'nf4'.
        return_int (bool, optional): whether return int data. Defaults to False.

    Returns:
        q_tensor: fake quantized tensor
    """
    assert data_type in FLOAT_MAPPING, "unexpected data type."
    allow_data = FLOAT_MAPPING[data_type]
    allow_data_bit = INT_MAPPING[data_type]
    # get scale and update tensor
    if scale is None:
        scale = tensor.abs().max(-1)[0] * quantile / max(allow_data)
        scale = scale.unsqueeze(dim=-1)
    tensor = tensor / scale
    mid_data = [(allow_data[i] + allow_data[i + 1]) / 2 for i in range(len(allow_data) - 1)]
    q_tensor = torch.zeros_like(tensor)
    for i in range(len(allow_data)):
        data = allow_data_bit[i] if return_int else allow_data[i]
        if i == 0:
            q_tensor += torch.where(tensor <= mid_data[i], data, 0)
        elif i == len(allow_data) - 1:
            q_tensor += torch.where(tensor > mid_data[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_data[i - 1] < tensor) & (tensor <= mid_data[i]), data, 0)
    if return_int:
        return q_tensor.type(torch.int8), scale.type(torch.float), None
    return q_tensor * scale


def qdq_weight_asym(weight, num_bits=4, quantile=1.0, return_int=False):
    """Quant and dequant tensor with asym schema.

    Args:
        weight:  input weight
        num_bits (int, optional): num_bits. Defaults to 4.
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.

    Returns:
        output: qdq weight
    """
    maxq = torch.tensor(2**num_bits - 1)
    zeros_shape = list(weight.shape[:-1])
    zeros = torch.zeros(*zeros_shape, device=weight.device)
    wmin = torch.minimum(weight.min(-1)[0], zeros)
    wmax = torch.maximum(weight.max(-1)[0], zeros)
    wmin = wmin * quantile
    wmax = wmax * quantile
    tmp = (wmin == 0) & (wmax == 0)
    wmin[tmp] = -1
    wmax[tmp] = +1
    scale = (wmax - wmin) / maxq
    zp = torch.round(-wmin / scale)
    scale.unsqueeze_(dim=-1)
    zp.unsqueeze_(dim=-1)
    q = torch.clamp(torch.round(weight / scale) + zp, 0, maxq)
    if return_int:
        return q.type(torch.uint8), scale.type(torch.float), zp.type(torch.uint8)
    return scale * (q - zp)


def qdq_weight_sym(weight, num_bits=4, quantile=1.0, return_int=False, full_range=False):
    """Quant and dequant tensor with sym schema.

    Args:
        weight : input weight
        num_bits (int, optional): num_bits. Defaults to 4.
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.
        full_range (bool, optional): Choose sym range whether use -2**(bits-1).
                For example: 4 bit
                    scale = amax / 8 if full_range else amax / 7
                    If True, scale = -scale if abs(min)> abs(max) else scale
                    Defaults to False.

    Returns:
        output: qdq weight
    """
    # assert num_bits > 1, "symmetric scheme only supports num_bits > 1"
    maxq = torch.tensor(2 ** (num_bits - 1) - 1).to(weight.device)
    minq = torch.tensor(-(2 ** (num_bits - 1))).to(weight.device)
    if num_bits == 1:  # pragma: no cover
        maxq = torch.tensor(2 ** (num_bits - 1))
        minq = torch.tensor(2 ** (num_bits - 1) - 1)
    max_val = torch.max(weight, -1)[0]
    min_val = torch.min(weight, -1)[0]
    flip_flag = torch.abs(max_val) > torch.abs(min_val)
    wmax = torch.max(torch.abs(max_val), torch.abs(min_val))
    wmax = wmax * quantile
    tmp = wmax == 0
    wmax[tmp] = +1
    if full_range:
        # use -8, 8 to make sure amax is not changed after fake quant
        scale = wmax / (-minq)
        tmp = scale * flip_flag.int()
        scale -= 2 * tmp  # set negetive scale with flip_flag
    else:
        scale = wmax / maxq
    scale.unsqueeze_(dim=-1)
    q = torch.clamp(torch.round(weight / scale), minq, maxq)
    if return_int:
        return q.type(torch.int8), scale.type(torch.float), None
    return scale * q


def qdq_weight_actor(weight, num_bits, scheme, quantile=1.0, data_type="int", return_int=False, full_range=False):
    """Quant and dequant tensor per channel.

    Args:
        weight : input weight
        num_bits (int, optional): num_bits. Defaults to 4.
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        data_type (str, optional): select from int, nf4, fp4. Defaults to int.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.
        full_range (bool, optional): Choose sym range whether use -2**(bits-1).

    Returns:
        output: qdq weight
    """
    assert num_bits > 0, "num_bits should be larger than 0"
    if "int" not in data_type and num_bits == 4:
        return quantize_4bit(weight, quantile=quantile, data_type=data_type, return_int=return_int)
    if scheme == "sym":
        return qdq_weight_sym(weight, num_bits, quantile, return_int, full_range)
    else:
        return qdq_weight_asym(weight, num_bits, quantile, return_int)


def quant_weight(
    weight, num_bits=4, group_size=-1, scheme="asym", quantile=1.0, data_type="int", return_int=False, full_range=False
):
    """Quant and dequant tensor with group size.

    Args:
        weight: input weight
        num_bits (int, optional): num_bits. Defaults to 4.
        group_size (int, optional): how many elements share one scale/zp. Defaults to -1.
        scheme (str, optional): sym or asym. Defaults to "asym".
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        data_type (str, optional): select from int, nf4, fp4. Defaults to int.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.
        full_range (bool, optional): Choose sym range whether use -2**(bits-1).

    Returns:
        output: qdq weight.
    """

    assert not return_int, "return_int is not supported in quant_weight"
    if num_bits <= 0:  # pragma: no cover
        return weight
    if group_size == -1 or weight.shape[-1] < group_size:
        return qdq_weight_actor(
            weight,
            num_bits,
            scheme=scheme,
            quantile=quantile,
            return_int=return_int,
            full_range=full_range,
            data_type=data_type,
        )

    orig_shape = weight.shape
    if weight.shape[-1] % group_size == 0:
        weight = weight.reshape(-1, group_size)
        weight = qdq_weight_actor(
            weight, num_bits, scheme=scheme, data_type=data_type, quantile=quantile, full_range=full_range
        )
        return weight.reshape(orig_shape)
    else:
        split_index = weight.shape[-1] // group_size * group_size
        weight1 = weight[:, :split_index]
        weight1 = weight1.reshape(-1, group_size)
        weight1 = qdq_weight_actor(
            weight1, num_bits, scheme=scheme, quantile=quantile, data_type=data_type, full_range=full_range
        )
        weight1 = weight1.reshape(orig_shape[0], split_index)
        weight2 = weight[:, split_index:]
        weight2 = qdq_weight_actor(
            weight2, num_bits, scheme=scheme, data_type=data_type, quantile=quantile, full_range=full_range
        )
        weight = torch.cat([weight1, weight2], dim=1)
        return weight


def search_clip(m, num_bits=4, group_size=32, scheme="asym", data_type="int", enable_full_range=False):
    """Search best clip range of each linears in current block.

    Args:
        m (torch.nn.Module): torch module.
        num_bits (int, optional): num bits.
        group_size (int, optional): how many elements share one scale/zp.
        scheme (str, optional): sym or asym.
        data_type (str, optional): select from int, nf4, fp4. Defaults to int.
        enable_full_range (bool, optional): Choose sym range whether use -2**(bits-1).

    Returns:
        best_clip_ratio (float): best percentile of clip
    """
    org_weight = m.weight.data
    logger.info("Searching the best clip range with RTN algorithm")
    best_error = float("inf")
    best_clip_ratio = None
    n_grid = 200
    max_shrink = 0.2
    history = []
    for i_s in range(int(max_shrink * n_grid)):
        ratio = 1 - i_s / n_grid  # 1, 0.805-1.0
        cur_weight = quant_weight(
            m.weight.data,
            num_bits=num_bits,
            group_size=group_size,
            scheme=scheme,
            data_type=data_type,
            full_range=enable_full_range,
            quantile=ratio,
        )
        loss = (org_weight - cur_weight).float().pow(2).mean().item()
        history.append(loss)
        is_best = loss < best_error
        if is_best:
            best_error = loss
            best_clip_ratio = ratio
    logger.debug("The loss history of different clip range:{}".format(history))
    logger.debug("The best clip ratio is {}".format(best_clip_ratio))
    return best_clip_ratio


def rtn_quantize(
    model,
    num_bits=4,
    group_size=32,
    scheme="asym",
    quantile=1.0,
    weight_config={},
    return_int=False,
    data_type="int",
    enable_full_range=False,
    enable_mse_search=False,
    group_dim=1,
    **kwargs,
):
    """Quant the model with round to nearst method.

    Args:
        model: torch module
        num_bits: num bits. Defaults to 4.
        group_size (int, optional): how many elements share one scale/zp. Defaults to 32.
        scheme (str, optional): sym or asym. Defaults to "asym".
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        data_type (str, optional): select from int, nf4, fp4. Defaults to int.
        weight_config (dict, optional): specific layer wise configurations. Defaults to {}.
            For example,
                weight_config={
                    'fc2':
                        {
                            'bits': 4,
                            'group_size': 32,
                            'scheme': 'sym'
                            'gptq_perm': [1, 1, ...] # for gptq perm
                        }
                }
        return_int (bool, optional): Choose return fp32 or int32 model.
                                     Defaults to False.
        enable_full_range (bool, optional): Choose sym range whether use -2**(bits-1).
                                     Defaults to False.
        enable_mse_search (bool, optional):  Whether search clip range.
                                     Defaults to True.
        group_dim (int, optional):   0 means splitting output channel,
                                     1 means splitting input channel. Defaults to 1.

    Returns:
        model: fake quantized torch module
    """
    assert isinstance(model, torch.nn.Module), "only support torch module"
    supported_layers = ["Linear"]
    if return_int:
        compression_dtype = kwargs.get("compression_dtype", torch.int32)
        compression_dim = kwargs.get("compression_dim", 1)
        scale_dtype = kwargs.get("scale_dtype", torch.float32)
        device = kwargs.get("device", "cpu")
        use_optimum_format = kwargs.get("use_optimum_format", True)
    for name, m in model.named_modules():
        if m.__class__.__name__ not in supported_layers:
            continue
        orig_dtype = next(m.parameters()).dtype
        if orig_dtype != torch.float:
            m = m.float()
        if name in weight_config:  # pragma: no cover
            num_bits = weight_config[name]["bits"]
            group_size = weight_config[name]["group_size"]
            scheme = weight_config[name]["scheme"]
            quantile = weight_config[name].get("quantile", 1.0)
        logger.debug(f"RTN quantized module:{name, m}")
        log_msg = (
            f"RTN quantization config: num_bits={num_bits}, group_size={group_size}, "
            + f"scheme={scheme}, quantile={quantile}"
        )
        if data_type != "int":
            log_msg += f", dtype={data_type}"
        elif scheme == "sym":  # nf4/fp4 is always [-7,7]
            log_msg += f", enable_full_range={enable_full_range}"
        logger.debug(log_msg)
        if num_bits <= 0:
            logger.info(f"Skip {name}")
            continue
        weight = m.weight.T if group_dim == 0 else m.weight
        if enable_mse_search:
            quantile = search_clip(m, num_bits, group_size, scheme, data_type, enable_full_range)
        if return_int:
            from .model_wrapper import WeightOnlyLinear

            int_weight, scale, zp = quant_weight(
                weight,
                num_bits,
                group_size,
                scheme,
                quantile,
                data_type=data_type,
                return_int=True,
                full_range=enable_full_range,
            )
            int_weight = int_weight.T if group_dim == 0 else int_weight
            scale = scale.T if group_dim == 0 else scale
            zp = zp.T if group_dim == 0 and zp is not None else zp
            new_module = WeightOnlyLinear(
                m.in_features,
                m.out_features,
                num_bits,
                group_size,
                dtype=data_type,
                zp=zp is not None,
                bias=m.bias is not None,
                compression_dtype=compression_dtype,
                compression_dim=compression_dim,
                scale_dtype=scale_dtype,
                device=device,
                use_optimum_format=use_optimum_format,
            )
            new_module.pack(int_weight, scale, zp, m.bias)
            if name == "":
                return new_module
            else:
                set_module(model, name, new_module)
        else:
            q_weight = quant_weight(
                weight,
                num_bits,
                group_size,
                scheme,
                quantile,
                data_type=data_type,
                full_range=enable_full_range,
            )
            q_weight = q_weight.T if group_dim == 0 else q_weight
            m.weight.data.copy_(q_weight)
        if orig_dtype != torch.float:
            m = m.to(orig_dtype)
    return model


def gptq_quantize(
    model,
    weight_config={},
    dataloader=None,
    nsamples=128,
    use_max_length=True,
    pad_max_length=2048,
    device=None,
    layer_wise=False,
    model_path=None,
):
    """Run weight-only quantization with."""
    # TODO: unify weight_config keys, add docstring, and support default config
    assert isinstance(model, torch.nn.Module), "only support torch module"
    if layer_wise:
        assert model_path is not None, "model_path should not be None when use layer_wise mode"
    from .gptq import GPTQuantizer

    gptq_quantizer = GPTQuantizer(
        model, weight_config, dataloader, nsamples, use_max_length, pad_max_length, device, layer_wise=layer_wise
    )
    fp32_modified_model, gptq_config = gptq_quantizer.execute_quantization(model_path=model_path)
    logger.info("GPTQ quantizing done.")
    return fp32_modified_model, gptq_config


@torch.no_grad()
def awq_quantize(
    model,
    bits=4,
    group_size=32,
    scheme="asym",
    weight_config={},
    example_inputs=None,
    dataloader=None,
    n_samples=128,
    calib_func=None,
    enable_auto_scale=True,
    enable_mse_search=True,
    folding=False,
    return_int=False,
    enable_full_range=False,
    data_type="int",
):
    """Quant the model with Activation-aware Weight quantization(AWQ) method.

    Args:
        model (torch.nn.Module): torch model.
        example_inputs: example_inputs.
        weight_config (dict, optional): contains all info required by AWQ. Defaults to {}.
            For example,
                weight_config={
                    'fc2':
                        {
                            # 'absorb_layer': 'fc1',
                            'bits': 4,
                            'group_size': 32,
                            'scheme': 'sym'
                        }
                }
        absorb_dict (dict, optional): contains all absorb info required by AWQ.. Defaults to {}.
            For example,
                absorb_dict = {
                    # 'absorb_layer': absorbed_layer
                    'fc1': ['fc1', 'fc2', 'fc3']
                } # in this case, fc2 and fc3 need to share the same scale. fc1 is self absorbed.
                # self absorb module will replace with MulLinear, which contains torch.mul and module.
        n_samples: calibration sample number.
        enable_auto_scale (bool, optional): whether enable scale for salient weight. Defaults to True.
        enable_mse_search (bool, optional):  whether enable clip for weight by checking mse. Defaults to True.
        calib_func: a custom inference function to replace dataloader and iters.
        n_blocks: split model into block number to avoid OOM.
        return_int (bool, optional): Choose return fp32 or int32 model.
                                     Defaults to False.
        enable_full_range (bool, optional): Choose sym range whether use -2**(bits-1).

    Returns:
        model: fake quantized model
    """
    from .awq import ActAwareWeightQuant

    assert isinstance(model, torch.nn.Module), "only support torch module"
    awq = ActAwareWeightQuant(
        model,
        example_inputs=example_inputs,
        calib_func=calib_func,
        dataloader=dataloader,
        n_samples=n_samples,
        bits=bits,
        group_size=group_size,
        scheme=scheme,
        enable_full_range=enable_full_range,
        weight_config=weight_config,
        data_type=data_type,
    )
    qdq_model = awq.quantize(
        enable_auto_scale=enable_auto_scale,
        enable_mse_search=enable_mse_search,
        folding=folding,
        return_int=return_int,
    )
    return qdq_model


def teq_quantize(
    model, weight_config={}, absorb_to_layer={}, extra_config={}, dataloader=None, calib_func=None, example_inputs=None
):
    """Run weight-only quantization with."""
    assert isinstance(model, torch.nn.Module), "only support torch module"
    logger.info("TEQ quantizing start.")
    if example_inputs is None:
        if dataloader is None:  # pragma: no cover
            assert False, "Please provide dataloader or example_inputs for TEQ algorithm."
        try:
            for idx, (input, label) in enumerate(dataloader):
                example_inputs = input
                break
        except:  # pragma: no cover
            for idx, input in enumerate(dataloader):
                example_inputs = input
                break

    from .teq import TEQuantizer

    teq_quantizer = TEQuantizer(model, weight_config, absorb_to_layer, extra_config, example_inputs)

    # 1. wrapper tuning scale to model
    teq_quantizer.add_tuning_scale()

    # 2. tuning
    # custom train function, there calls calib_func
    if calib_func:  # pragma: no cover
        calib_func(teq_quantizer.model)
    else:
        if dataloader is None:  # pragma: no cover
            assert False, "Please provide dataloader to train."
        teq_quantizer.train(dataloader)

    # 3. apply scale to model
    teq_quantizer.transform()

    # 4. get quantized model
    teq_quantizer.quantize()

    # quantization_data = gptq_quantizer.execute_quantization()
    logger.info("TEQ quantizing done.")
    return teq_quantizer.model


def quant_weight_w_scale(weight, scale, zp, group_size=-1):
    """Quant and dequant tensor with group size.

    Args:
        weight: input weight
        scale: scale
        zp: zero point
        group_size (int, optional): how many elements share one scale/zp. Defaults to -1.

    Returns:
        output: int weight.
    """
    device = weight.device
    scale = scale.to(device)
    if zp is not None:
        zp = zp.to(device)
    if group_size == -1:
        return torch.round(weight / scale) if zp is None else torch.round(weight / scale + zp)
    int_weight = torch.zeros(weight.shape).to(device)
    leng = weight.shape[1] // group_size
    tail_flag = False if weight.shape[1] % group_size == 0 else True
    for i in range(leng):
        int_weight_tmp = weight[:, i * group_size : (i + 1) * group_size] / scale[:, i].unsqueeze(1)
        if zp is not None:
            int_weight_tmp += zp[:, i].unsqueeze(1)
        int_weight[:, i * group_size : (i + 1) * group_size] = torch.round(int_weight_tmp)
    if tail_flag:
        int_weight_tmp = weight[:, leng * group_size :] / scale[:, -1].unsqueeze(1)
        if zp is not None:
            int_weight_tmp += zp[:, -1].unsqueeze(1)
        int_weight[:, leng * group_size :] = torch.round(int_weight_tmp)
    return int_weight
