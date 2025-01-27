# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import argparse
import os
import glob

from imaginaire.config import Config
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.dataset import get_test_dataloader
from imaginaire.utils.distributed import init_dist
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.utils.io import get_checkpoint as get_checkpoint
from imaginaire.utils.trainer import \
    (get_model_optimizer_and_scheduler, get_trainer, set_random_seed)
import imaginaire.config
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', required=False, 
                        help='Path to the training config file.')
    parser.add_argument('--checkpoint', default='',
                        help='Checkpoint path.')
    parser.add_argument('--output_dir', required=False,
                        help='Location to save the image outputs, default is in the logdir given by --logdir.')
    parser.add_argument('--logdir', required=True,
                        help='Location to save the log files, default is logs.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--style_std', type=float, default=1.0)
    parser.add_argument('--save_raw_output', action='store_true')  # saves only grid plot if false
    parser.add_argument('--num_images', type=int, default=10000)
    args = parser.parse_args()
    return args

""" 

CUDA_VISIBLE_DEVICES=3 python inference.py --single_gpu --save_raw_output --style_std 0.65 \
    --logdir logs/cs2cs-higher_gen_lr_lower_res/2023_0525_2220_42_ampO1_lower_LR_really_lower_res \
        --checkpoint logs/cs2cs-higher_gen_lr_lower_res/2023_0525_2220_42_ampO1_lower_LR_really_lower_res/checkpoints/epoch_00039_iteration_000400000_checkpoint.pt \
        

CUDA_VISIBLE_DEVICES=3 python inference.py --single_gpu --save_raw_output --style_std 2.0 \
    --logdir logs/cs2cs_make_complex_arch/2023_0524_2217_59_ampO1_lower_LR_arch_experiments \
        --checkpoint logs/cs2cs_make_complex_arch/2023_0524_2217_59_ampO1_lower_LR_arch_experiments/checkpoints/epoch_00040_iteration_000400000_checkpoint.pt \
        

"""

def main():
    args = parse_args()
    set_affinity(args.local_rank)
    set_random_seed(args.seed, by_rank=True)

    # get config from logdir
    if args.config is None:
        # find the config, it should be in the logdir, and should be the only .yaml file
        configs = glob.glob(os.path.join(args.logdir, '*.yaml'))
        assert len(configs) == 1, f'Found {len(configs)} configs in {args.logdir}, expected 1'
        args.config = configs[0]
        print(f'Using config {args.config}')
    
    cfg = Config(args.config)
    imaginaire.config.DEBUG = args.debug

    # if output_dir is not specified, use logdir
    # the folder name will include the epoch, iteration, and style_std
    if args.output_dir is None:
        # first get iteration from cp name, example name: epoch_00040_iteration_000400000_checkpoint
        cp_name = os.path.basename(args.checkpoint)
        iteration = cp_name.split('_')[3]
        args.output_dir = os.path.join(args.logdir, 'inference', f'inference_cp_{iteration}_style_std_{args.style_std}')
        # create the output dir
        os.makedirs(args.output_dir, exist_ok=True)
        print(f'Using output_dir {args.output_dir}')

    if not hasattr(cfg, 'inference_args'):
        cfg.inference_args = None

    # If args.single_gpu is set to True,
    # we will disable distributed data parallel.
    if not args.single_gpu:
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank)

    # Override the number of data loading workers if necessary
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    # # Create log directory for storing training results.
    # cfg.date_uid, cfg.logdir = init_logging(args.config, args.logdir)

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.
    test_data_loader = get_test_dataloader(cfg)
    net_G, net_D, opt_G, opt_D, sch_G, sch_D = \
        get_model_optimizer_and_scheduler(cfg, seed=args.seed)
    trainer = get_trainer(cfg, net_G, net_D,
                          opt_G, opt_D,
                          sch_G, sch_D,
                          None, test_data_loader)

    if args.checkpoint == '':
        # Download pretrained weights.
        pretrained_weight_url = cfg.pretrained_weight
        if pretrained_weight_url == '':
            logging.info('link to the pretrained weight is not specified.')
            raise
        default_checkpoint_path = args.config.split('.yaml')[0] + '-' + cfg.pretrained_weight + '.pt'
        args.checkpoint = get_checkpoint(default_checkpoint_path, pretrained_weight_url)
        logging.info('Checkpoint downloaded to', args.checkpoint)

    # Load checkpoint.
    trainer.load_checkpoint(cfg, args.checkpoint)

    # Do inference.
    trainer.current_epoch = -1
    trainer.current_iteration = -1
    trainer.test(test_data_loader, args.output_dir, cfg.inference_args, style_std=args.style_std, save_raw_output=args.save_raw_output)


if __name__ == "__main__":
    main()
