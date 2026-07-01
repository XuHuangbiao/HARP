import os, sys

import torch

from utils import misc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import yaml
import argparse
from utils.misc import str2bool


class Parser(object):
    """Args parser"""

    def __init__(self):

        self.get_args()

        self.setup()

        self.check_args()



    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--benchmark', type=str, choices=['BEST', 'EPIC_Skills', 'CoffeeCraft'], help='dataset', default='BEST')
        parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
        parser.add_argument('--fix_bn', type=str2bool, default=True)
        parser.add_argument('--resume', type=str2bool, default=False,
                            help='autoresume training from exp dir (interrupted by accident)')
        parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='device')

        parser.add_argument('--use_score', type=str2bool, default=False, help='')

        parser.add_argument('--split', type=int, default=None,
                            help='Specify split number, e.g., 1 will generate train_split1.txt and test_split1.txt')
        parser.add_argument('--device', type=list, default=[0], help='device')
        parser.add_argument('--seed', type=int, default=42, help='random seed')
        parser.add_argument('--phase', type=str, default='test', help='train or test')
        parser.add_argument('--config', type=str, default=None, help='config file')
        parser.add_argument('--bs_train', type=int, default=None, help='batchsize of train')
        parser.add_argument('--bs_test', type=int, default=None, help='batchsize of test')
        parser.add_argument('--num_groups', type=int, default=4, help='number of group size')
        parser.add_argument('--task', type=str, default='tie_tie', help="task's name used to training")
        parser.add_argument('--frame_label_path', type=str, default=None, help=None)
        parser.add_argument('--pretrained_i3d_weight', type=str, default='weights/model_rgb.pth')
        self.args = parser.parse_args()

    def setup(self):
        if self.args.config is None:
            self.args.config = f'configs/{self.args.benchmark}.yaml'
        else:
            self.args.config = f'configs/{self.args.config}.yaml'

        self.get_config()

        self.merge_config()

        self.args.data_root = os.path.join(self.args.data_root, self.args.task)

        self.args.feature_path = os.path.join(self.args.data_root, 'features')
        self.args.frame_path = os.path.join(self.args.data_root, 'frames')
        self.args.video_path = os.path.join(self.args.data_root, 'videos')

        if self.args.split is not None:
            train_file = f'train_split{self.args.split}.txt'
            test_file = f'test_split{self.args.split}.txt'
        else:
            train_file = 'train.txt'
            test_file = 'test.txt'

        self.args.train_pair_path = os.path.join(self.args.data_root, 'info', train_file)
        self.args.test_pair_path = os.path.join(self.args.data_root, 'info', test_file)

        if self.args.phase != 'train':
            self.args.is_training = False

    def get_config(self):
        print(f'----------------------------\n'
              f'Load yaml from {self.args.config}.\n'
              f'----------------------------\n')

        with open(self.args.config) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

    def merge_config(self):
        for k, v in self.config.items():
            if k not in vars(self.args).keys():
                setattr(self.args, k, v)
            elif vars(self.args)[k] == None:
                setattr(self.args, k, v)



    def check_args(self):
        if (self.args.benchmark == 'BEST' or self.args.benchmark == 'EPIC_Skills'
                or self.args.benchmark == 'CoffeeCraft'):
            self.args.use_skill_data = True
        else:
            self.args.use_skill_data = False

        if self.args.use_skill_data:
            if self.args.split is not None:
                task_with_split = f"{self.args.task}/split_{self.args.split}"
            else:
                task_with_split = self.args.task
            self.args.exp_path = os.path.join('./exps', self.args.benchmark, task_with_split, self.args.exp_name)
        else:
            self.args.exp_path = os.path.join('./exps', self.args.benchmark, self.args.exp_name)


        if self.args.use_gpu:
            self.args.device = [int(i) for i in self.args.device]
            self.args.output_device = self.args.device[0]

        misc.init_seed(self.args.seed)

        if self.args.resume:
            cfg_path = os.path.join(self.args.exp_path, f'configs/{self.args.benchmark}.yaml')
            print(f'----------------------------\n'
                  f'Resume yaml from {cfg_path}.\n'
                  f'----------------------------\n')
