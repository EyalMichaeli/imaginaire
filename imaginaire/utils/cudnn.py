# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch.backends.cudnn as cudnn
import logging
from imaginaire.utils.distributed import master_only_print as print


def init_cudnn(deterministic, benchmark):
    r"""Initialize the cudnn module. The two things to consider is whether to
    use cudnn benchmark and whether to use cudnn deterministic. If cudnn
    benchmark is set, then the cudnn deterministic is automatically false.

    Args:
        deterministic (bool): Whether to use cudnn deterministic.
        benchmark (bool): Whether to use cudnn benchmark.
    """
    cudnn.deterministic = deterministic
    cudnn.benchmark = benchmark
    logging.info('cudnn benchmark: {}'.format(benchmark))
    logging.info('cudnn deterministic: {}'.format(deterministic))
