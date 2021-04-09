import argparse

import torch
import ipdb

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-grid', type=int, default=2,
                        help='num grid of direct_control and indirect_control' )
    parser.add_argument('--random-noise-frame', action='store_true',
                         help='if add a random noise to frame')
    parser.add_argument('--epsilon', type=float,
                         help='epsilon for random-noise-frame')
    parser.add_argument('--latent-control-intrinsic-reward-type', type=str,
                        help='M/G/delta_uG/__binary/NONE__relu/NONE__sum/hash_count_bouns/__clip_G/NONE' )
    parser.add_argument('--latent-control-discount', type=float,
                        help='G map of latent control discount' )
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.obs_size = 84
    try:
        args.crop_obs = {
            "PongNoFrameskip-v4": {
                'h': [14,args.obs_size  ],
                'w': [0 ,args.obs_size  ],
            },
            "BreakoutNoFrameskip-v4": {
                'h': [14,args.obs_size  ],
                'w': [0 ,args.obs_size  ],
            },
        }[args.env_name]
    except Exception as e:
        args.crop_obs = None
        print('# WARNING: args.crop_obs = None')

    try:
        # in_channels, out_channels, kernel_size, stride
        args.model_structure = {
            4: {
                'DirectControlModel': {
                    # 84/4 = 21
                    'conv_0': ('X', 8, 5, 2),
                    # (21-5)/2+1 = 9
                    'conv_1': (8, 16, 4, 1),
                    # (9-4)/1+1 = 6
                    'conved_shape': (16, 6, 6),
                    'linear_size': 64,
                },
                'LatentControlModel': {
                    # 84/4 = 21
                    'conv_0': ('X', 16, 5, 2),
                    # (21-5)/2+1 = 9
                    'conv_1': (16, 32, 4, 1),
                    # (9-4)/1+1 = 6
                    'conved_shape': (32, 6, 6),
                    'linear_size': 1024,
                    'deconv_0': (32, 16, 4, 1),
                    # (6−1)×1+4 = 9
                    'deconv_1': (16, 1, 5, 2),
                    # (9−1)×2+5 = 21
                },
            },
            2: {
                'DirectControlModel': {
                    # 84/7 = 12
                    'conv_0': ('X', 1, 1, 1),
                    # (12-5)/1+1 = 8
                    'conv_1': (1, 1, 1, 1),
                    # (8-4)/1+1 = 5
                    'conved_shape': (1, 1, 1),
                    'linear_size': 64,
                },
                'LatentControlModel': {
                    # 84/7 = 12
                    'conv_0': ('X', 1, 1, 1),
                    # (12-5)/1+1 = 8
                    'conv_1': (1, 1, 1, 1),
                    # (8-4)/1+1 = 5
                    'conved_shape': (1, 1, 1),
                    'linear_size': 1024,
                    'deconv_0': (1, 1, 1, 1),
                    # (5−1)×1+4 = 8
                    'deconv_1': (1, 1, 1, 1),
                    # (8−1)×1+5 = 12
                },
            },
        }[args.num_grid]
    except Exception as e:
        input('# ACTION REQUIRED: args.crop_obs = None')

    return args
