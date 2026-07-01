import logging
import os, sys

import torch
from torch import nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

from utils import misc
from utils import feature_augments
import traceback


def dataset_builder(args):
    try:
        Dataset = misc.import_class("datasets." + args.benchmark)
        train_trans, test_trans = feature_augments.get_feature_trans(args)
        train_dataset = Dataset(args, transform=train_trans, subset='train')
        test_dataset = Dataset(args, transform=test_trans, subset='test')
        return train_dataset, test_dataset
    except Exception as e:
        traceback.print_exc()
        exit()

def train_dataloader_builder(args):
    train_dataset, _ = dataset_builder(args)
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.bs_train,
        shuffle=True,
        pin_memory=True,
        num_workers=int(args.workers),
        worker_init_fn=misc.worker_init_fn
    )
def test_dataloader_builder(args):
    _, test_dataset = dataset_builder(args)
    return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.bs_test,
            shuffle=False,
            pin_memory=True,
            num_workers=int(args.workers),
        )

def dae_builder(args):
    DAE = misc.import_class("models.dae")
    dae = DAE(args)
    return dae
def i3d_builder(args):
    Backbone = misc.import_class('models.I3D_backbone')
    backbone = Backbone(I3D_ckpt_path=args.pretrained_i3d_weight)
    return backbone

def model_builder(args):
    Model = misc.import_class("models.model")
    model = Model(args)
    return model


def logger_builder(self):
    
    logger = logging.getLogger(f"Processor_{self.args.exp_name}")
    logger.setLevel(logging.INFO)

    
    if logger.hasHandlers():
        logger.handlers.clear()

    
    log_sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(message)s', "%Y-%m-%d %H:%M:%S")
    log_sh.setFormatter(formatter)
    logger.addHandler(log_sh)

    
    log_dir = os.path.join(self.args.exp_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{self.args.phase}.log')
    if (self.args.resume and self.args.phase == 'train') or self.args.phase == 'test':
        log_fh = logging.FileHandler(log_file, mode='a')
    else:
        log_fh = logging.FileHandler(log_file, mode='w')
    log_fh.setLevel(logging.DEBUG)
    log_fh.setFormatter(formatter)
    logger.addHandler(log_fh)

    return logger


class Orthogonal_Loss_Builder(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(Orthogonal_Loss_Builder, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        total_loss = 0.0

        for layer_output in x:
            batch_size, num_vectors, vector_dim = layer_output.shape
            if num_vectors == 1:
                continue

            norm = torch.norm(layer_output, p=2, dim=2, keepdim=True)
            norm = torch.clamp(norm, min=self.epsilon)
            normalized_output = layer_output / norm  

            similarity_matrix = torch.matmul(normalized_output, normalized_output.transpose(1, 2))  
            identity_matrix = torch.eye(num_vectors, device=layer_output.device).unsqueeze(0)  

            diff = similarity_matrix - identity_matrix
            ortho_loss = torch.sum(diff ** 2, dim=[1, 2])  
            total_loss = total_loss + ortho_loss.mean()

        return total_loss

class MVP_Loss_Builder(nn.Module):
    def __init__(self, args):
        super(MVP_Loss_Builder, self).__init__()
        self.args = args
        alpha_init = args.loss_param['mvp_alpha']
        beta_init = args.loss_param['mvp_beta']
        gamma_init = args.loss_param['mvp_gamma']
        self.alpha = nn.Parameter(torch.log(torch.tensor(alpha_init)))
        self.beta = nn.Parameter(torch.log(torch.tensor(beta_init)))
        
        
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        self.margin_loss = Dynamic_Margin_Loss(m_init=args.loss_param['m3'], args=args, get_loss=False)

    def forward(self, mu_1, std_1, mu_2, std_2, epsilon=1e-8):
        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)
        gamma = F.sigmoid(self.gamma)

        sigma = torch.sqrt(std_1 ** 2 + std_2 ** 2 + epsilon)
        diff = (mu_1 - mu_2) / (sigma * (2 ** 0.5))
        prob = 0.5 * (1 + torch.erf(diff))
        prob = torch.clamp(prob, min=epsilon)

        ranking_loss = -torch.log(prob)
        m3 = self.margin_loss(mu_1, mu_2)
        diff_mu = torch.relu(m3 - (mu_1 - mu_2))

        
        large_variance_penalty = torch.relu(torch.log(gamma + sigma ** 2)).mean()  
        prob_mean_loss = ((ranking_loss + diff_mu) / (sigma ** 2)).mean()
        loss = alpha * prob_mean_loss + beta * large_variance_penalty
        return loss, m3

class Dynamic_Margin_Loss(nn.Module):
    def __init__(self, m_init=1.0, args=None, get_loss=True):
        super(Dynamic_Margin_Loss, self).__init__()
        alpha_init = args.loss_param['dym_alpha']
        beta_init = args.loss_param['dym_beta']
        self.alpha = torch.nn.Parameter(torch.log(torch.tensor(alpha_init)))
        self.beta = torch.nn.Parameter(torch.log(torch.tensor(beta_init)))
        self.m = m_init
        self.get_loss = get_loss

    def forward(self, s1, s2):
        diff = s1 - s2
        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)
        margin = alpha * F.tanh(diff * beta) + self.m
        if not self.get_loss:
            return margin
        margin_loss = torch.relu(margin - diff)
        loss = margin_loss.mean()
        return loss, margin