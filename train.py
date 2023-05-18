# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import argparse
import os
import sys
import random
import logging
import torch.autograd.profiler as profiler
import wandb
import datetime
from pathlib import Path

import imaginaire.config
from imaginaire.config import Config
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.dataset import get_train_and_val_dataloader
from imaginaire.utils.distributed import init_dist, is_master, get_world_size
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.utils.misc import slice_tensor
# from imaginaire.utils.logging import make_logging_dir
from imaginaire.utils.trainer import (get_model_optimizer_and_scheduler,
                                      get_trainer, set_random_seed)


sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))



def get_date_uid():
    """Generate a unique id based on date.
    Returns:
        str: Return uid string, e.g. '20171122171307111552'.
    """
    return str(datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S"))


def init_logging(config_path, logdir):
    r"""Create log directory for storing checkpoints and output images.

    Args:
        config_path (str): Path to the configuration file.
        logdir (str): Log directory name
    Returns:
        str: Return log dir
    """
    root_dir = 'logs'
    config_file = os.path.basename(config_path)
    date_uid = get_date_uid()
    # example: logs/2019_0125_1047_58_spade_cocostuff
    
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    log_folder_name = '_'.join([date_uid, os.path.splitext(config_file)[0]])
    log_folder = os.path.join(logdir, log_folder_name)
    os.makedirs(log_folder, exist_ok=True)

    log_file = os.path.join(log_folder, 'log.log')
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(fh)
    print('Log file: {}'.format(log_file))
    logging.info('Log directory: {}'.format(log_folder))
    logging.info('Config file: {}'.format(config_path))
    return date_uid, log_folder

# def init_logging(config_path, logdir):
#     r"""Create log directory for storing checkpoints and output images.
#     Args:
#         config_path (str): Path to the configuration file.
#         logdir (str): Log directory name
#     Returns:
#         str: Return log dir
#     """
#     config_file = os.path.basename(config_path)
#     root_dir = 'logs'
#     date_uid = get_date_uid()
#     # example: logs/2019_0125_1047_58_spade_cocostuff
#     log_folder_name = '_'.join([date_uid, os.path.splitext(config_file)[0]])
#     os.makedirs(root_dir, exist_ok=True)
#     log_folder = os.path.join(root_dir, log_folder_name)
#     os.makedirs(log_folder, exist_ok=True)
#     log_file = os.path.join(root_dir, log_folder_name, 'log.log')
#     logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
#     fh = logging.FileHandler(log_file, mode='w')
#     fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
#     logging.getLogger().addHandler(fh)
#     if logdir is None:
#         logdir = os.path.join(root_dir, log_folder_name)
#     return date_uid, logdir


def main():
    args = parse_args()

    # Create log directory for storing training results.
    date_uid, logdir = init_logging(args.config, args.logdir)
    # make_logging_dir(cfg.logdir)

    set_affinity(args.local_rank)
    if args.randomized_seed:
        args.seed = random.randint(0, 10000)
    set_random_seed(args.seed, by_rank=True)
    cfg = Config(args.config)
    cfg.date_uid, cfg.logdir = date_uid, logdir
    
    # try:
    #     from userlib.auto_resume import AutoResume
    #     AutoResume.init()
    # except:  # noqa
    #     pass

    # If args.single_gpu is set to True,
    # we will disable distributed data parallel
    if not args.single_gpu:
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank)
    logging.info(f"Training with {get_world_size()} GPUs.")

    # Global arguments.
    imaginaire.config.DEBUG = args.debug
    imaginaire.config.USE_JIT = args.use_jit

    # Override the number of data loading workers if necessary
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    # copy the config given by the user to the log dir
    os.system(f"cp {args.config} {cfg.logdir}")

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.
    batch_size = cfg.data.train.batch_size
    total_step = max(cfg.trainer.dis_step, cfg.trainer.gen_step)
    cfg.data.train.batch_size *= total_step
    train_data_loader, val_data_loader = get_train_and_val_dataloader(cfg, args.seed)
    net_G, net_D, opt_G, opt_D, sch_G, sch_D = \
        get_model_optimizer_and_scheduler(cfg, seed=args.seed)
    # print the LR
    for param_group in opt_G.param_groups:
        logging.info(f"LR of G: {param_group['lr']}")
    trainer = get_trainer(cfg, net_G, net_D,
                          opt_G, opt_D,
                          sch_G, sch_D,
                          train_data_loader, val_data_loader)
    resumed, current_epoch, current_iteration = trainer.load_checkpoint(cfg, args.checkpoint, args.resume)

    # Initialize Wandb.
    if is_master():
        if args.wandb_id is not None:
            wandb_id = args.wandb_id
        else:
            if resumed and os.path.exists(os.path.join(cfg.logdir, 'wandb_id.txt')):
                with open(os.path.join(cfg.logdir, 'wandb_id.txt'), 'r+') as f:
                    wandb_id = f.read()
            else:
                wandb_id = wandb.util.generate_id()
                with open(os.path.join(cfg.logdir, 'wandb_id.txt'), 'w+') as f:
                    f.write(wandb_id)
        wandb_mode = "disabled" if (args.debug or not args.wandb) else "online"
        wandb.init(id=wandb_id,
                   project=args.wandb_name,
                   config=cfg,
                   name=os.path.basename(cfg.logdir),
                   resume="allow",
                   settings=wandb.Settings(start_method="fork"),
                   mode=wandb_mode)
        wandb.config.update({'dataset': cfg.data.name})
        wandb.watch(trainer.net_G_module)
        wandb.watch(trainer.net_D.module)

    logging.info('Starting training...')
    # Start training.
    for epoch in range(current_epoch, cfg.max_epoch):
        logging.info('Epoch {} ...'.format(epoch))
        if not args.single_gpu:
            train_data_loader.sampler.set_epoch(current_epoch)
        trainer.start_of_epoch(current_epoch)
        for it, data in enumerate(train_data_loader):
            with profiler.profile(enabled=args.profile,
                                  use_cuda=True,
                                  profile_memory=True,
                                  record_shapes=True) as prof:
                data = trainer.start_of_iteration(data, current_iteration)

                for i in range(cfg.trainer.dis_step):
                    trainer.dis_update(
                        slice_tensor(data, i * batch_size,
                                     (i + 1) * batch_size))
                for i in range(cfg.trainer.gen_step):
                    trainer.gen_update(
                        slice_tensor(data, i * batch_size,
                                     (i + 1) * batch_size))

                current_iteration += 1
                trainer.end_of_iteration(data, current_epoch, current_iteration)
                if current_iteration >= cfg.max_iter:
                    logging.info('Done with training!!!')
                    return
            if args.profile:
                logging.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                prof.export_chrome_trace(os.path.join(cfg.logdir, "trace.json"))
            # try:
            #     if AutoResume.termination_requested():
            #         trainer.save_checkpoint(current_epoch, current_iteration)
            #         AutoResume.request_resume()
            #         logging.info("Training terminated. Returning")
            #         return 0
            # except:  # noqa
            #     logging.info("AutoResume didn't work")
            #     pass

        current_epoch += 1
        trainer.end_of_epoch(data, current_epoch, current_iteration)
    logging.info('Done with training!!!')
    return


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config',
                        help='Path to the training config file.', required=True)
    parser.add_argument('--logdir', help='Dir for saving logs and models.')
    parser.add_argument('--checkpoint', default='', help='Checkpoint path.')
    parser.add_argument('--seed', type=int, default=2, help='Random seed.')
    parser.add_argument('--randomized_seed', action='store_true', help='Use a random seed between 0-10000.')
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_jit', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_name', default='default', type=str)
    parser.add_argument('--wandb_id', type=str)
    parser.add_argument('--resume', type=int)
    parser.add_argument('--num_workers', type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

"""
commands:

nohup sh -c 'CUDA_VISIBLE_DEVICES=2 python train.py --logdir logs/cs2cs-default_run --config /mnt/raid/home/eyal_michaeli/git/imaginaire/configs/projects/munit/cs2cs/ampO1_lower_LR.yaml --single_gpu' 2>&1 | tee -a /mnt/raid/home/eyal_michaeli/git/imaginaire/cs2cs-default_run.log &

nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python train.py --logdir logs/cs2cs-style_recon_2_perceptual_1 --config /mnt/raid/home/eyal_michaeli/git/imaginaire/configs/projects/munit/cs2cs/ampO1_lower_LR.yaml --single_gpu' 2>&1 | tee -a /mnt/raid/home/eyal_michaeli/git/imaginaire/cs2cs-style_recon_2_perceptual_1.log &

nohup sh -c 'CUDA_VISIBLE_DEVICES=1 python train.py --logdir logs/cs2cs_style_recon_2_instead_of_1 --config /mnt/raid/home/eyal_michaeli/git/imaginaire/configs/projects/munit/cs2cs/ampO1_lower_LR.yaml --single_gpu' 2>&1 | tee -a /mnt/raid/home/eyal_michaeli/git/imaginaire/cs2cs_style_recon_2_instead_of_1.log &

resume:
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python train.py --resume 1 --checkpoint logs/2023_0413_2053_12_ampO1_lower_LR/checkpoints/epoch_00002_iteration_000200000_checkpoint.pt --config /mnt/raid/home/eyal_michaeli/git/imaginaire/configs/projects/munit/bdd10k2bdd10k/ampO1_lower_LR.yaml --single_gpu' 2>&1 | tee -a /mnt/raid/home/eyal_michaeli/git/imaginaire/munit_bdd2bdd_v0_continue_200k.log &

low LR config:
nohup sh -c 'CUDA_VISIBLE_DEVICES=3 python train.py --config /mnt/raid/home/eyal_michaeli/git/imaginaire/configs/projects/munit/bdd10k2bdd10k/ampO1_lower_LR.yaml --single_gpu' 2>&1 | tee -a /mnt/raid/home/eyal_michaeli/git/imaginaire/munit_bdd2bdd_v0.log &

"""



"""
env installation:

in order to install env, i:
1. installed pytroch 1.9
2. installed req.txt
3. installed third parties using (sh scripts/install_with_conda.sh  - I commented out the installing env part in the script)
    IMPORTANT: ONLY did that after running export CUDA_HOME=/usr/local/cuda-11.1 to make sure that one is visible.
"""