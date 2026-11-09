import os
import torch
import warnings
import numpy as np
import math
import torch.nn as nn
warnings.filterwarnings('ignore')
from tqdm import tqdm
from utils.metrics import metric
from thop import profile, clever_format
import time
import gc

class Engine_Forecasting(object):
    def __init__(self, args):
        self.args = args
        self.data_id = args.data_id + '_' + str(args.seq_len) + '_' + str(args.pred_len)
        self.criterion = torch.nn.MSELoss()
        self.alpha= args.train.alpha

    def train_batch(self, batch, model, optimizer, batch_num):
        model.train()
        optimizer.zero_grad()
        max_buffer_size = 5
        all_loss = 0
        num_loaders = len(batch)
        node_num = batch[0].shape[1]
        x, start_days , start_intervals, poi, satellite, loc, adj, y, mean, std, state ,dataset,topk_indices, topk_values= [
            t.to(self.args.device) if isinstance(t, torch.Tensor) else t for t in batch
        ]
        assert not torch.isnan(y).any(), "y contains NaN"
        assert not torch.isinf(x).any(), "x contains INF"
        with self._autocast():
            y_hat, multimodal, multimodal1, multimodal2, balance_loss = model(x,start_days,start_intervals, poi, satellite, loc, adj, mean, std, state, topk_indices, topk_values)
        if self.args.is_norm:
            mean_std = np.load(os.path.join(self.args.data.root_path, "data","mean_std.npy"),allow_pickle=True).item() 
            mean,std=mean_std[dataset][0],mean_std[dataset][1]
            y_hat = y_hat * std + mean
            y = y * std + mean
        pre_loss = model.get_loss(y, y_hat) 
        if multimodal1 is not None:
            con_loss = model.get_contrastive_loss(multimodal, multimodal1, multimodal2)
        else:
            con_loss = torch.tensor(0.0)
        
        if balance_loss is not None:
            balance_loss=10 * (balance_loss)
        else:
            balance_loss = torch.tensor(0.0)
        total_loss = (pre_loss + 0.1*con_loss + balance_loss)   
        total_loss.backward() 
        optimizer.step()
        lr_now = optimizer.param_groups[0]['lr']
        all_loss = total_loss.item()    
        con_loss=con_loss.item()
        return all_loss, pre_loss.item(), con_loss, float(balance_loss), lr_now
    

    def _autocast(self):
        # bf16 autocast only on CUDA; plain fp32 on CPU
        return torch.autocast(device_type=self.args.device.type, dtype=torch.bfloat16,
                              enabled=self.args.device.type == 'cuda')

    def _denorm(self, a, dataset):
        if self.args.is_norm:
            mean_std = np.load(os.path.join(self.args.data.root_path, "data", "mean_std.npy"), allow_pickle=True).item()
            m, s = mean_std[dataset][0], mean_std[dataset][1]
            a = a * s + m
        return a

    def _evaluate(self, loader, model, desc):
        """Streaming version of utils.metrics.metric (same definitions) so the full
        prediction tensor never has to sit in RAM."""
        model.eval()
        n = 0
        s_abs = s_sq = s_ape = s_smape = 0.0
        n_ape = 0
        losses = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                x, start_days, start_intervals, poi, satellite, loc, adj, y, mean, std, state, dataset, topk_indices, topk_values = [
                    t.to(self.args.device) if isinstance(t, torch.Tensor) else t for t in batch]
                with self._autocast():
                    y_hat, _, _, _, _ = model(x, start_days, start_intervals, poi, satellite, loc, adj, mean, std, state, topk_indices, topk_values)
                y_hat = y_hat.float()
                losses.append(model.get_loss(y, y_hat).item())
                p = self._denorm(y_hat.double().cpu(), dataset)
                t = self._denorm(y.double().cpu(), dataset)
                err = p - t
                n += err.numel()
                s_abs += err.abs().sum().item()
                s_sq += (err ** 2).sum().item()
                m = t > 1e-3
                s_ape += (err[m].abs() / t[m]).sum().item()
                n_ape += int(m.sum())
                den = p.abs() + t.abs()
                sm = 200.0 * err.abs() / den.clamp(min=1e-8)
                sm[(t.abs() < 1e-8) & (p.abs() < 1e-8)] = 0.0
                s_smape += sm.sum().item()
        mae = s_abs / n
        rmse = (s_sq / n) ** 0.5
        mape = 100.0 * s_ape / max(n_ape, 1)
        smape = s_smape / n
        self.args.logger.info('Setting: {}, RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}, SMAPE:{:.6f}'.format(self.data_id, rmse, mae, mape, smape))
        return float(np.mean(losses)), mae, rmse, mape, smape

    def valid(self, valid_loader, model):
        return self._evaluate(valid_loader, model, 'valid')

    def test(self, test_loader, model):
        _, mae, rmse, mape, smape = self._evaluate(test_loader, model, 'test')
        f = open(os.path.join(self.args.checkpoint, 'result_s' + str(self.args.seed) + '.txt'), 'a')
        f.write(self.data_id + '\n')
        f.write('RMSE: {}, MAE: {}, MAPE: {}, SMAPE: {}'.format(rmse, mae, mape, smape))
        f.write('\n\n')
        f.close()
