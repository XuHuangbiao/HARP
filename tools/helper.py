import logging
import os, sys
import shutil

from scipy.stats import stats
from torch import optim
import numpy as np
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch


def save_files(self):
    
    args_dict = vars(self.args)

    config_dir = os.path.join(self.args.exp_path, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    config_file = os.path.join(config_dir, f'{self.args.benchmark}.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(args_dict, f)

    self.logger.info(f'Save args to {config_file}.')

    
    if self.args.phase == 'train':
        shutil.copytree('./models', os.path.join(self.args.exp_path, 'models'), dirs_exist_ok=True)
        shutil.copytree('./tools', os.path.join(self.args.exp_path, 'tools'), dirs_exist_ok=True)
        shutil.copytree('./utils', os.path.join(self.args.exp_path, 'utils'), dirs_exist_ok=True)

        self.logger.info(f'Back-up copy models, tools, and utils to {self.args.exp_path}.')


def save_checkpoint(self, epoch, exp_name, param_dict=None, use_skill_data=False, use_score=False):
    
    checkpoint_dir = os.path.join(self.args.exp_path, 'weights')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, exp_name + '.pth')

    
    checkpoint_data = {
        'model': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'epoch': epoch,
    }

    if use_score:
        
        checkpoint_data.update({
            'rho': param_dict.get('rho', None),
            'L2': param_dict.get('L2', None),
            'RL2': param_dict.get('RL2', None),
            'epoch_best': self.epoch_best,
            'rho_best': self.rho_best,
            'L2_min': self.L2_min,
            'RL2_min': self.RL2_min,
        })

    if use_skill_data:
        
        checkpoint_data.update({
            'prec': param_dict.get('prec', None),
            'prec_best': self.prec_best,  
        })

    
    torch.save(checkpoint_data, checkpoint_file)
    self.logger.info(f'Save checkpoint to {checkpoint_file}.')

def save_best(self, epoch, param_dict=None, use_skill_data=False, use_score=False, pred_scores=None,
              true_scores=None):

    
    self.L2_min = param_dict.get('L2', self.L2_min)
    self.RL2_min = param_dict.get('RL2', self.RL2_min)
    self.prec_best = param_dict.get('prec', self.prec_best)
    self.rho_best = param_dict.get('rho', self.rho_best)
    self.epoch_best = epoch

    
    self.logger.info('----- New best found -----')
    self.print_best()

    self.save_checkpoint(epoch, 'best', param_dict, use_skill_data=use_skill_data, use_score=use_score)

    
    if pred_scores is not None and true_scores is not None:
        save_path_pred = os.path.join(self.args.exp_path, 'weights/pred.npy')
        save_path_true = os.path.join(self.args.exp_path, 'weights/true.npy')
        np.save(save_path_pred, pred_scores)
        np.save(save_path_true, true_scores)


def save_history(self):
        history_file = os.path.join(self.args.exp_path, 'weights/history.npz')
        np.savez(history_file, train=self.history['train'], test=self.history['test'])

        self.logger.info(f'Save history to {history_file}.')

def print_best(self):
    self.logger.info(f' Best epoch: {self.epoch_best + 1:d}')
    if self.args.use_skill_data:
        self.logger.info(f'Precision: {self.prec_best:.4f}')

    if self.args.use_score:
        self.logger.info(f'Correlation: {self.rho_best:.4f}')
        self.logger.info(f'         L2: {self.L2_min:.4f}')
        self.logger.info(f'        RL2: {self.RL2_min:.4f}')

def build_opti_sche(self):
    self.logger.info(f'Build optimizer and scheduler...')

    if self.args.optimizer == 'Adam':
        self.optimizer = optim.Adam(
            [
                {'params': self.model.parameters()},
                {'params': self.ranking_loss.parameters()},
                {'params': self.ranking_aware_loss.parameters()},
                {'params': self.mvp_loss.parameters()},
                {'params': self.uni_margin_loss.parameters()}
            ],
            lr=self.args.base_lr,
            weight_decay=self.args.weight_decay
        )
    elif self.args.optimizer == 'AdamW':
        self.optimizer = optim.AdamW(
            [
                {'params': self.model.parameters()},
                {'params': self.ranking_loss.parameters()},
                {'params': self.ranking_aware_loss.parameters()},
                {'params': self.mvp_loss.parameters()},
                {'params': self.uni_margin_loss.parameters()}
            ],
            lr=self.args.base_lr,
            weight_decay=self.args.weight_decay
        )
    else:
        raise NotImplementedError()

    
    self.lr_scheduler = None

def compute_matric(self, pred_scores, true_scores):
    rho, p = stats.spearmanr(pred_scores, true_scores)

    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)

    L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
    RL2 = 100 * np.power((pred_scores - true_scores) /
                   (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]

    return rho, p, L2, RL2


def analyse_result(self, epoch, pred_scores, true_scores, prec, subset='TRAIN'):
    if self.args.use_skill_data:
        
        if self.args.use_score:
            
            rho, p, L2, RL2 = self.compute_matric(pred_scores, true_scores)
            self.history['test'].append((epoch, prec, rho, p, L2, RL2))

            self.logger.info(
                f' [{subset} {epoch + 1:d}] Precision: {prec:.4f} ({self.prec_best:.4f}), '
                f' Correlation: {rho:.4f}, '
                f'L2: {L2:.4f}, RL2: {RL2:.4f}')

            self.update_params(rho, L2, RL2, prec)
            return
        else:
            
            self.history['test'].append((epoch, prec))  
            if subset == 'TRAIN':
                self.logger.info(f' [{subset} {epoch + 1:d}] Precision: {prec : .4f} ')
            else:
                self.logger.info(f' [{subset} {epoch + 1:d}] Precision: {prec : .4f} '
                                 f'( best prec: {self.prec_best:.4f} in epoch {self.epoch_best + 1} )')

            self.update_params(None, None, None, prec)
            return

    else:
        
        rho, p, L2, RL2 = self.compute_matric(pred_scores, true_scores)  
        self.history['test'].append((epoch, rho, p, L2, RL2))
        self.logger.info(f' [{subset} {epoch + 1:d}] Correlation: {rho:.4f} ({self.rho_best:.4f}), '
                         f'L2: {L2:.4f}, RL2: {RL2:.4f}')
        self.update_params(rho, L2, RL2, None)
        return

def compute_true_score(self, good_name, bad_name):
    
    good_score = list(map(self.task_scores.get, good_name))
    bad_score = list(map(self.task_scores.get, bad_name))

    
    good_score_array = np.array(good_score, dtype=np.float32)
    bad_score_array = np.array(bad_score, dtype=np.float32)

    
    return good_score_array - bad_score_array


def load_model_weights(model, checkpoint_path, logger):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['model'])

    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return model