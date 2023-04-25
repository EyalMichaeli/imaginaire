# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import datetime
import os
import logging

from imaginaire.utils.distributed import master_only
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.meters import set_summary_writer



@master_only
def make_logging_dir(logdir):
    r"""Create the logging directory

    Args:
        logdir (str): Log directory name
    """
    print('Make folder {}'.format(logdir))
    os.makedirs(logdir, exist_ok=True)
    tensorboard_dir = os.path.join(logdir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    cp_dir = os.path.join(logdir, 'checkpoints')
    os.makedirs(cp_dir, exist_ok=True)
    # set_summary_writer(tensorboard_dir)
