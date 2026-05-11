import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pypose as pp

torch.set_printoptions(sci_mode=True, precision=6)


def gaussian_init_weights(m, mean=0.0, std=0.02):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.normal_(m.weight, mean=mean, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0) 

def zero_conv_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.constant_(m.weight, 0.0)
        if m.bias is not None:
            nn.init.constant_(m.bias,   0.0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
def init_gru(m):
    if isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
                
class IMUNet(nn.Module):
    def __init__(self, prop_cov=True, device='cuda:0'):
        super().__init__()
        self.prop_cov = prop_cov
        self.device = device
        self.B = 0
        self.T = 0
        self.D = 0
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=32, kernel_size=10, stride=5, padding=0),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.gru1 = nn.GRU(input_size=32,  hidden_size=64, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True)

        self.accdecoder   = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3))
        self.gyrdecoder   = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3))
        self.acccov_decoder = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3))
        self.gyrcov_decoder = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3))

        self.acc_std = torch.tensor(0.1, dtype=torch.float32).to(device)
        self.gyr_std = torch.tensor(np.pi/180, dtype=torch.float32).to(device)
        
    def encoder(self, x, valid_length):
        x = x.to(next(self.cnn.parameters()).device)
        
        x = self.cnn(x.transpose(-1, -2))
        x = x.transpose(-1, -2)
        
        valid_feat_length = ((valid_length - 2) - 2)

        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x
    
    def cov_decoder(self, x):
        acc_cov = torch.exp(self.acccov_decoder(x) - 5.)
        gyr_cov = torch.exp(self.gyrcov_decoder(x) - 5.)
        return acc_cov, gyr_cov
    
    def noise_decoder(self, x):
        acc_noise = self.accdecoder(x) * self.acc_std
        gyr_noise = self.gyrdecoder(x) * self.gyr_std
        return acc_noise, gyr_noise
    
    def inference(self, data, valid_length):
        acc = data['accels']
        gyr = data['gyros']
        
        
        x = torch.cat([acc, gyr], dim=-1)
        feat = self.encoder(x, valid_length)
        acc_noise, gyr_noise = self.noise_decoder(feat)
        
        if self.prop_cov is True:
            acc_cov, gyr_cov = self.cov_decoder(feat)
        else:
            acc_cov, gyr_cov = None, None
        return acc_noise, gyr_noise, acc_cov, gyr_cov
    
    def broadcast_to_valid(self, orig, update, valid_length, mode='add'):
        result = orig.clone()
        outs = []
        for b in range(self.B):
            vlen = valid_length[b].item()
            seg_len = vlen // self.D
            remainder = vlen % self.D

            start = 0
            for d in range(self.D):
                end = start + seg_len + (1 if d < remainder else 0)
                if start >= vlen:
                    break
                end = min(end, vlen)
                if mode == 'add':
                    result[b, start:end] += update[b, d]
                elif mode == 'assign':
                    result[b, start:end] = update[b, d]
                start = end
            outs.append(result[b, :vlen].clone())
        return outs
    
    def forward(self, data):
        data['accels'] = data['accels'].to(self.device).to(torch.float32)
        data['gyros'] = data['gyros'].to(self.device).to(torch.float32)
        data['imu_dts'] = data['imu_dts'].to(self.device)
        valid_length = data['valid_length']

        correction_acc, correction_gyr, cov_acc, cov_gyr = self.inference(data, valid_length)

        self.B, self.T, _ = data['accels'].shape
        self.D = correction_acc.shape[1]

        valid_acc = self.broadcast_to_valid(data['accels'], correction_acc, valid_length, mode='add')
        valid_gyr = self.broadcast_to_valid(data['gyros'], correction_gyr, valid_length, mode='add')
        valid_cov_acc = self.broadcast_to_valid(torch.zeros_like(data['accels']), cov_acc, valid_length, mode='assign') if cov_acc is not None else None
        valid_cov_gyr = self.broadcast_to_valid(torch.zeros_like(data['gyros']), cov_gyr, valid_length, mode='assign') if cov_gyr is not None else None
        valid_dts = [data['imu_dts'][b, :valid_length[b]] for b in range(self.B)]
        return {
            'accels_corr': valid_acc,
            'gyros_corr': valid_gyr,
            'acc_cov': valid_cov_acc,
            'gyr_cov': valid_cov_gyr,
            'valid_length': valid_length,
            'dts': valid_dts,
        }