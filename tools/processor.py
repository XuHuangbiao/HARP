import os, sys
import shutil
import time

import numpy as np
import yaml
from scipy import stats
from torch import optim, nn
from torch.distributions import kl
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

from tools import builder, helper
from utils import misc
import torch
from thop import profile
class Processor(object):

    def __init__(self, args):
        self.args = args

        self.logger = builder.logger_builder(self)

        self.save_files()

        self.load_data()

        self.load_models()

        self.load_loss()

        self.build_opti_sche()

        self.load_pretrain()

        self.param_dict = {
            'rho': 0,
            'L2': 1000,
            'RL2': 1000,
            'prec': 0
        }


    def update_params(self, rho=None, L2=None, RL2=None, prec=None):
        if rho is not None:
            self.param_dict['rho'] = rho
        if L2 is not None:
            self.param_dict['L2'] = L2
        if RL2 is not None:
            self.param_dict['RL2'] = RL2
        if prec is not None:
            self.param_dict['prec'] = prec


    def load_data(self):
        self.dataloader = {}

        if self.args.phase == 'train':
            self.dataloader['train'] = builder.train_dataloader_builder(self.args)
        self.dataloader['test'] = builder.test_dataloader_builder(self.args)


    def load_pretrain(self):
        self.history = {}
        self.history['train'] = []
        self.history['test'] = []

        if self.args.resume is not True and self.args.phase == 'train':
            
            self.start_epoch = 0
            self.epoch_best = 0
            self.rho_best = 0
            self.L2_min, self.RL2_min = 1000, 1000
            self.prec_best = 0

            return
        return

    def load_loss(self):
        self.ranking_loss = builder.Dynamic_Margin_Loss(m_init=self.args.loss_param['m1'], args=self.args)
        self.uni_margin_loss = builder.Dynamic_Margin_Loss(m_init=self.args.loss_param['m1'], args=self.args)
        self.ranking_aware_loss = builder.Dynamic_Margin_Loss(m_init=self.args.loss_param['m2'], args=self.args)
        self.mvp_loss = builder.MVP_Loss_Builder(self.args)
        self.orth_loss = builder.Orthogonal_Loss_Builder()

        if self.args.use_gpu:
            self.uni_margin_loss = self.uni_margin_loss.to(self.args.output_device)
            self.ranking_loss = self.ranking_loss.to(self.args.output_device)
            self.orth_loss = self.orth_loss.to(self.args.output_device)
            self.mvp_loss = self.mvp_loss.cuda(self.args.output_device)

    def load_models(self):
        
        self.model = builder.model_builder(self.args)

        
        if self.args.use_gpu:
            self.model = self.model.cuda(self.args.output_device)

            if len(self.args.device) > 1:
                self.model = misc.use_DataParallel(self.args, self.model)

                self.logger.info(f'{len(self.args.device)} GPUs available, using DataParallel.')

    def get_accuracy(self, pred_scores):
        """Computes the % of correctly ordered pairs (based on pred_scores > 0)"""
        pred_scores = np.array(pred_scores, dtype=np.float32)  
        correct = pred_scores > 0
        return float(correct.sum()) / len(correct)

    def save_files(self):
        helper.save_files(self)

    def save_checkpoint(self, epoch, exp_name, param_dict=None, use_skill_data=False, use_score=False):
        helper.save_checkpoint(self, epoch, exp_name, param_dict, use_skill_data, use_score)

    def save_best(self, epoch, param_dict=None, use_skill_data=False, use_score=False, pred_scores=None, true_scores=None):
        helper.save_best(self, epoch, param_dict, use_skill_data, use_score, pred_scores, true_scores)



    def save_history(self):
        helper.save_history(self)

    def print_best(self):
        helper.print_best(self)

    def build_opti_sche(self):
        helper.build_opti_sche(self)

    def compute_matric(self, pred_scores, true_scores):
        return helper.compute_matric(self, pred_scores, true_scores)


    def analyse_result(self, epoch, pred_scores, true_scores, prec, subset='TRAIN'):
        helper.analyse_result(self, epoch, pred_scores, true_scores, prec, subset)

    def compute_true_score(self, good_name, bad_name):
        return helper.compute_true_score(self, good_name, bad_name)

    def safe_squeeze(self, tensor, dim=-1):
        if tensor.dim() > 1:
            tensor = tensor.squeeze(dim)

        if tensor.shape[dim] > 1:
            tensor = tensor.squeeze(dim)
        return tensor

    def compute_ranking_loss_one_uni(self, pred_1=None, pred_2=None, uni_1=None, uni_2=None, target=None, mu_1=None, std_1=None, mu_2=None,
                             std_2=None, all_leaf_outs1=None, all_leaf_outs2=None):

        ranking_loss, m11 = self.ranking_loss(pred_1, pred_2)
        loss = ranking_loss

        mvp_loss, m3 = self.mvp_loss(mu_1, std_1, mu_2, std_2)
        loss += mvp_loss

        uni_loss, m12 = self.ranking_loss(uni_1, uni_2)
        loss += uni_loss

        diff_pred = pred_1 - pred_2
        diff_uni = (uni_1 - uni_2).detach()
        r_w_loss, m2 = self.ranking_aware_loss(diff_pred, diff_uni)
        loss += r_w_loss

        orth_loss = 1.0 * (self.orth_loss(all_leaf_outs1) + self.orth_loss(all_leaf_outs2))
        loss += orth_loss
        self.L_orth.append(orth_loss.detach())

        self.current_loss.append(loss.detach())
        self.L_mvp.append(mvp_loss.detach())
        self.m11.append(m11.detach())
        self.m12.append(m12.detach())
        self.m2.append(m2.detach())
        self.m3.append(m3.detach())

        return loss

    
    def eval_step(self, epoch):
        self.model.eval()

        self.args.is_training = False

        loader = self.dataloader['test']

        true_scores = []
        pred_scores = []

        prec = None

        process = tqdm(range(len(loader)), dynamic_ncols=True)
        with torch.no_grad():
            for batch_idx, ((good_feature, bad_feature), (good_name, bad_name)) in enumerate(loader):

                if self.args.use_gpu:
                    good_feature = good_feature.cuda(self.args.output_device)
                    bad_feature = bad_feature.cuda(self.args.output_device)

                
                refined_good_feature = good_feature
                refined_bad_feature = bad_feature

                
                pred_1, mu_1, std_1, uni_pred_1, orth_list = self.model(refined_good_feature)
                pred_2, mu_2, std_2, uni_pred_2, orth_list = self.model(refined_bad_feature)
                pred = pred_1 - pred_2

                
                pred_scores.append(pred.cpu().detach())

                
                process.set_description(f'(BS {self.args.bs_test})')
                process.update()

            process.close()

            pred_scores = torch.cat(pred_scores).numpy().tolist()

            
            if self.args.use_skill_data:
                prec = self.get_accuracy(pred_scores)

            
            self.analyse_result(epoch, pred_scores, true_scores, prec, 'EVAL')

            if self.args.phase == 'train':
                
                self.save_checkpoint(epoch, 'last', self.param_dict, self.args.use_skill_data, self.args.use_score)

                
                if self.args.use_skill_data and self.param_dict.get('prec') > self.prec_best:
                    self.save_best(epoch, self.param_dict, self.args.use_skill_data, self.args.use_score, pred_scores,
                                   true_scores)

                elif not self.args.use_skill_data and self.param_dict.get('rho') > self.rho_best:
                    self.save_best(epoch, self.param_dict, self.args.use_skill_data, self.args.use_score, pred_scores,
                                   true_scores)

    
    def train_step(self, epoch):
        self.model.train()

        self.args.is_training = True

        true_scores = []
        pred_scores = []

        loader = self.dataloader['train']

        process = tqdm(loader, dynamic_ncols=True)

        prec = None

        
        for batch_idx, ((good_feature, bad_feature), (good_name, bad_name)) in enumerate(process):

            target = torch.ones(good_feature.size(0), 1)

            
            if self.args.use_gpu:
                good_feature = good_feature.cuda(self.args.output_device)
                bad_feature = bad_feature.cuda(self.args.output_device)
                target = target.cuda(self.args.output_device)

            
            refined_good_feature = good_feature
            refined_bad_feature = bad_feature

            
            pred_1, mu_1, std_1, uni_pred_1, all_leaf_outs1 = self.model(refined_good_feature)
            pred_2, mu_2, std_2, uni_pred_2, all_leaf_outs2 = self.model(refined_bad_feature)
            pred = pred_1 - pred_2

            
            loss = self.compute_ranking_loss_one_uni(pred_1, pred_2, uni_pred_1, uni_pred_2, target,
                                                     mu_1, std_1, mu_2, std_2, all_leaf_outs1, all_leaf_outs2)

            
            
            

            
            pred_scores.append(pred.cpu().detach())

            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            
            process.set_description(f'(BS {self.args.bs_train}) loss: {loss.item():.2f}')
            process.update()

        
        pred_scores = torch.cat(pred_scores).numpy().tolist()
        process.close()

        
        if self.args.use_skill_data:
            prec = self.get_accuracy(pred_scores)

        
        self.analyse_result(epoch, pred_scores, true_scores, prec, 'TRAIN')

        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def safe_mean(self, tensor_list):
        """对 list[tensor] 求均值，支持不同形状的张量"""
        if not tensor_list:  
            return 0.0
        tensor_list = [t.flatten() for t in tensor_list]  
        return torch.cat(tensor_list).mean().item()  


    def start(self):
        for epoch in range(self.args.start_epoch, self.args.max_epoch):
            self.args.current_epoch = epoch
            self.current_loss = []
            self.L_mvp = []
            self.m11 = []
            self.m12 = []
            self.m2 = []
            self.m3 = []
            self.L_orth = []

            if self.lr_scheduler is not None:
                self.logger.info(f'+-------------------------------------------'
                                 f'[EPOCH: {epoch + 1}]--|--[lr: {self.lr_scheduler.get_last_lr()[0]:.6f}]'
                                 f'-------------------------------------------+')
            else:
                self.logger.info(f'+-------------------------------------------'
                                 f'[EPOCH: {epoch + 1}]'
                                 f'-------------------------------------------+')

            self.train_step(epoch)
            self.eval_step(epoch)

            self.logger.info(f'loss: {self.safe_mean(self.current_loss):.4f} '
                             f'| mvp: {self.safe_mean(self.L_mvp):.4f} '
                             f'| orth: {self.safe_mean(self.L_orth):.4f} '
                             f'| m11: {self.safe_mean(self.m11):.4f} '
                             f'| m12: {self.safe_mean(self.m12):.4f} '
                             f'| m2: {self.safe_mean(self.m2):.4f} '
                             f'| m3: {self.safe_mean(self.m3):.4f} ')

            
            

        self.logger.info(f'---------------- Processing over --------------------')
        self.print_best()
        self.save_history()
        self.logger.info(f'------------------------------------')

        return self.prec_best

    def i3d_test(self):
        from i3d_processor import i3d_process
        i3d_process(self)

    import torch

    def analyze_clip_importance(self, model_path, video_feature):
        """
        输入：模型路径 + 特征（(1, 400, 1024)）
        输出：每个 clip 的重要性（用梯度法，归一化后保证 > 0，clip 特征全0的重要性强制为0）
        """
        self.model.eval()
        device = torch.device('cuda' if self.args.use_gpu else 'cpu')

        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model']
        self.model.load_state_dict(state_dict)
        self.model.to(device)

        video_feature = video_feature.to(device)
        video_feature.requires_grad_(True)

        
        pred_score, mu, std, uni_pred, all_leaf_outs = self.model(video_feature)
        if pred_score.dim() > 1:
            pred_score = pred_score.squeeze()
        self.model.zero_grad()
        pred_score.backward()

        gradients = video_feature.grad.detach().cpu().squeeze(0)  
        clip_importance = torch.norm(gradients, dim=1)  

        
        zero_mask = (video_feature.detach().cpu().squeeze(0) == 0).all(dim=1)

        
        min_val = clip_importance.min()
        clip_importance = clip_importance - min_val
        max_val = clip_importance.max()
        if max_val > 1e-6:
            clip_importance /= max_val
        else:
            clip_importance = torch.zeros_like(clip_importance)

        
        clip_importance[zero_mask] = 0.0

        valid_mask = ~zero_mask
        clip_importance = clip_importance[valid_mask]

        return clip_importance.numpy()

    def latte(self, model_path):
        self.model.eval()
        device = torch.device('cuda' if self.args.use_gpu else 'cpu')

        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model']
        self.model.load_state_dict(state_dict)
        self.model.to(device)

        self.args.is_training = False

        loader = self.dataloader['test']

        pred_scores = []  

        
        amateur_vs_amateur_count = 0
        amateur_vs_amateur_correct = 0
        amateur_vs_pro_count = 0
        amateur_vs_pro_correct = 0
        pro_vs_amateur_count = 0
        pro_vs_amateur_correct = 0
        pro_vs_pro_count = 0
        pro_vs_pro_correct = 0

        
        import numpy as np
        stage_path = "/home/yh/data/backup/a/skillaqa/data/CoffeeCraft/latte_art_stage_dict.npy"
        try:
            stage_dict = np.load(stage_path, allow_pickle=True).item()
            print(f"[Stage Info] 已載入 {len(stage_dict)} 個樣本的階段分類 (1=業餘, 0=專業)")
        except Exception as e:
            print(f"[Stage Info] 載入 stage_dict 失敗: {e}")
            stage_dict = {}  

        process = tqdm(range(len(loader)), dynamic_ncols=True)

        with torch.no_grad():
            for batch_idx, ((good_feature, bad_feature), (good_name, bad_name)) in enumerate(loader):

                if self.args.use_gpu:
                    good_feature = good_feature.cuda(self.args.output_device)
                    bad_feature = bad_feature.cuda(self.args.output_device)

                
                refined_good_feature = good_feature
                refined_bad_feature = bad_feature

                
                pred_1, mu_1, std_1, uni_pred_1, _ = self.model(refined_good_feature)
                pred_2, mu_2, std_2, uni_pred_2, _ = self.model(refined_bad_feature)
                pred = pred_1 - pred_2  

                pred_scores.append(pred.cpu().detach())

                
                if stage_dict:
                    
                    pred = pred.squeeze()  
                    if pred.dim() != 1:
                        print(f"警告：pred 維度異常 {pred.shape}，請檢查模型輸出")

                    pred_list = pred.cpu().tolist()  

                    for g_name, b_name, score in zip(good_name, bad_name, pred_list):
                        
                        if isinstance(score, list):
                            score = score[0] if score else 0.0

                        g_stage = stage_dict.get(g_name, 0)
                        b_stage = stage_dict.get(b_name, 0)

                        is_correct = score > 0

                        if g_stage == 1 and b_stage == 1:
                            amateur_vs_amateur_count += 1
                            if is_correct: amateur_vs_amateur_correct += 1
                        elif g_stage == 1 and b_stage == 0:
                            amateur_vs_pro_count += 1
                            if is_correct: amateur_vs_pro_correct += 1
                        elif g_stage == 0 and b_stage == 1:
                            pro_vs_amateur_count += 1
                            if is_correct: pro_vs_amateur_correct += 1
                        else:
                            pro_vs_pro_count += 1
                            if is_correct: pro_vs_pro_correct += 1

                
                process.set_description(f'(BS {self.args.bs_test})')
                process.update()

        process.close()

        
        pred_scores = torch.cat(pred_scores).numpy().tolist()

        
        if stage_dict:
            print("\n" + "═" * 80)
            print("CoffeeCraft Latte Art 模型測試 - 不同階段對戰正確率統計")
            print(" (業餘=amateur=1, 專業=pro=0) ".center(80))
            print("═" * 80)

            groups = [
                ("業餘 vs 業餘", amateur_vs_amateur_count, amateur_vs_amateur_correct),
                ("業餘 vs 專業", amateur_vs_pro_count, amateur_vs_pro_correct),
                ("專業 vs 業餘", pro_vs_amateur_count, pro_vs_amateur_correct),
                ("專業 vs 專業", pro_vs_pro_count, pro_vs_pro_correct),
            ]

            total_count = 0
            total_correct = 0

            for group_name, count, correct in groups:
                acc = (correct / count * 100) if count > 0 else float('nan')
                print(f"{group_name:<12} : {count:6d} 對    正確 {correct:5d}    → {acc:6.2f}%")
                total_count += count
                total_correct += correct

            total_acc = (total_correct / total_count * 100) if total_count > 0 else float('nan')
            print("-" * 80)
            print(f"總          計 : {total_count:6d} 對    正確 {total_correct:5d}    → {total_acc:6.2f}%")
            print("═" * 80 + "\n")


    def get_orth(self, model_path, video_feature):
        """
        输入：模型路径 + 特征（(1, 400, 1024)）
        输出：每个 clip 的重要性（用梯度法，归一化后保证 > 0，clip 特征全0的重要性强制为0）
        """
        self.model.eval()
        device = torch.device('cuda' if self.args.use_gpu else 'cpu')

        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model']
        self.model.load_state_dict(state_dict)
        self.model.to(device)

        video_feature = video_feature.to(device)

        
        with torch.no_grad():
            pred_score, mu, std, uni_pred, all_leaf_outs = self.model(video_feature)

        return all_leaf_outs

    def compute_flops_params(self):
        device = torch.device("cuda" if self.args.use_gpu else "cpu")

        

        self.model = builder.model_builder(self.args).to(device)
        self.model.eval()
        torch.set_grad_enabled(False)
        
        seq_len = self.args.models['seq_len']
        dim = 1024  

        dummy_input = torch.randn(1, seq_len, dim).to(device)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        flops, params = profile(self.model, inputs=(dummy_input,))
        
        self.model.eval()
        with torch.no_grad():
            
            for _ in range(10):
                _ = self.model(dummy_input)
            torch.cuda.synchronize()  

            total_time = 0.0
            num_runs = 100
            for _ in range(num_runs):
                start = time.time()
                _ = self.model(dummy_input)
                torch.cuda.synchronize()  
                total_time += (time.time() - start) * 1000

        avg_time_ms = total_time / num_runs

        
        print(f"Average Inference Time over {num_runs} runs: {avg_time_ms:.3f} ms")
        print(f"Model Params: {params / 1e6:.3f} million")
        print(f"Model GFLOPs: {flops / 1e9:.3f} billion")

    def tsne(self, model_path="/home/yh/data/backup/a/skillaqa/exps/CoffeeCraft/latte_art/FIRST/weights/best.pth"):
        self.model.eval()
        device = torch.device('cuda' if self.args.use_gpu else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)

        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint  

        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict, strict=False)  
        self.model.to(device)

        print(f"模型加载完成，设备：{device}，参数量：{sum(p.numel() for p in self.model.parameters())}")

        return self.model

