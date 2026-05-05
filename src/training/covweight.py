import torch
import torch.nn.functional as F

class CovWeightController:
    def __init__(self, ema_beta=0.9, target_ratio=0.2, warmup_steps=1000,
                 min_w=0.01, max_w=0.5, device="cpu"):
        self.beta = ema_beta
        self.alpha = target_ratio
        self.warmup_steps = warmup_steps
        self.min_w = min_w
        self.max_w = max_w
        self.device = torch.device(device)
        self._reset()

    def _reset(self):
        d = self.device
        self.ema_state_r = torch.tensor(0.0, device=d)
        self.ema_cov_r   = torch.tensor(1.0, device=d)
        self.ema_state_v = torch.tensor(0.0, device=d)
        self.ema_cov_v   = torch.tensor(1.0, device=d)
        self.ema_state_t = torch.tensor(0.0, device=d)
        self.ema_cov_t   = torch.tensor(1.0, device=d)
        
        # Component usage tracking
        self.component_usage = {}
        self.total_updates = 0

    @torch.no_grad()
    def update(self, rot_loss, vel_loss, pos_loss,
               rot_cov_loss, vel_cov_loss, pos_cov_loss, global_step: int, 
               comp_ids=None, comp_weights=None):
        b = self.beta
        # EMA 업데이트
        self.ema_state_r = b*self.ema_state_r + (1-b)*rot_loss.detach()
        self.ema_cov_r   = b*self.ema_cov_r   + (1-b)*rot_cov_loss.detach()
        self.ema_state_v = b*self.ema_state_v + (1-b)*vel_loss.detach()
        self.ema_cov_v   = b*self.ema_cov_v   + (1-b)*vel_cov_loss.detach()
        self.ema_state_t = b*self.ema_state_t + (1-b)*pos_loss.detach()
        self.ema_cov_t   = b*self.ema_cov_t   + (1-b)*pos_cov_loss.detach()

        eps = 1e-8
        # state 대비 cov 비중을 alpha로 맞춤
        w_r = torch.clamp(self.alpha * self.ema_state_r/(self.ema_cov_r+eps),
                          self.min_w, self.max_w)
        w_v = torch.clamp(self.alpha * self.ema_state_v/(self.ema_cov_v+eps),
                          self.min_w, self.max_w)
        w_t = torch.clamp(self.alpha * self.ema_state_t/(self.ema_cov_t+eps),
                          self.min_w, self.max_w)

        # 워밍업
        m = min(1.0, float(global_step+1)/float(self.warmup_steps))
        
        # Track component usage if comp_ids and comp_weights are provided
        if comp_ids is not None and comp_weights is not None:
            # Convert to numpy if needed
            if hasattr(comp_ids, 'cpu'):
                comp_ids_np = comp_ids.cpu().numpy()
            else:
                comp_ids_np = comp_ids
            if hasattr(comp_weights, 'cpu'):
                comp_weights_np = comp_weights.cpu().numpy()
            else:
                comp_weights_np = comp_weights
            
            # Track usage for each component
            for comp_id, weight in zip(comp_ids_np, comp_weights_np):
                if weight > 0:  # Only count actually used components
                    if comp_id not in self.component_usage:
                        self.component_usage[comp_id] = 0
                    self.component_usage[comp_id] += 1
                    self.total_updates += 1
        
        return m*w_r, m*w_v, m*w_t
    
    def get_component_stats(self):
        """Get component usage statistics"""
        return {
            'component_usage': self.component_usage.copy(),
            'total_updates': self.total_updates,
            'unique_components': len(self.component_usage)
        }
