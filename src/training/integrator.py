import torch
import pypose as pp


def prase_init(init=None, motion_mode=False, device='cuda:0'):
    dtype = torch.get_default_dtype()

    if init is not None:
        if motion_mode:
            init_pos = torch.zeros(3, dtype=dtype).to(device)
            init_rot = pp.SO3(init['rot']).to(dtype).to(device)
            init_vel = torch.zeros(3, dtype=dtype).to(device)
        else:
            init_pos = init['pos'].to(dtype).to(device)
            init_rot = pp.SO3(init['rot']).to(dtype).to(device)
            init_vel = init['vel'].to(dtype).to(device)
    else:
        init_pos = torch.zeros(3, dtype=dtype).to(device)
        init_rot = pp.identity_SO3().to(dtype).to(device)
        init_vel = torch.zeros(3, dtype=dtype).to(device)

    if 'cov' not in init or init['cov'] is None:
        init_cov = torch.eye(9, dtype=torch.get_default_dtype(), device=device).unsqueeze(0) * 1e-10
    else:
        init_cov = init['cov']

    return init_pos, init_rot, init_vel, init_cov 

class IMUIntegrator:
    def __init__(self, init_state=None, prop_cov=True, gravity=torch.tensor([0.0, 0.0, 9.81]), device='cuda:0'):
        self.device = device
        init_pos, init_rot, init_vel, _ = prase_init(init_state, motion_mode=False)
        self.integrator = pp.module.IMUPreintegrator(init_pos, init_rot, init_vel,
                                                     prop_cov=prop_cov, reset=True, gravity=gravity).to(device)
    
    def integrate(self, init, dts, accels, gyros, cov_accels=None, cov_gyros=None, motion_mode=False, device='cuda:0'):
        init_pos, init_rot, init_vel, init_cov = prase_init(init, motion_mode, device)

        if motion_mode:
            poses, rots, covs, vels = [], [], [], []
        else:
            poses = [init_pos.cpu()]
            rots  = [init_rot.cpu()]
            covs  = [init_cov.cpu()]
            vels  = [init_vel.cpu()]

        last_state = init

        for b in range(len(dts)):
            dt   = dts[b].unsqueeze(0).clone().unsqueeze(-1)
            # print(dt)
            gyro = gyros[b].unsqueeze(0).clone()
            acc  = accels[b].unsqueeze(0).clone()
            if cov_accels is not None:
                cov_acc  = cov_accels[b].unsqueeze(0).clone()
                cov_gyr  = cov_gyros[b].unsqueeze(0).clone()

                state = self.integrator(dt=dt, gyro=gyro, acc=acc,
                                        acc_cov=cov_acc, gyro_cov=cov_gyr,
                                        init_state=last_state)
            else:
                state = self.integrator(dt=dt, gyro=gyro, acc=acc,
                                        init_state=last_state)

            covs.append(state['cov'].cpu())
            poses.append(state['pos'][..., -1, :].squeeze().cpu())
            vels.append(state['vel'][..., -1, :].squeeze().cpu())

            if motion_mode:
                rel_rot = last_state['rot'].Inv() @ state['rot'][..., -1, :].squeeze()
                rots.append(rel_rot.cpu())
            else:
                rots.append(state['rot'][..., -1, :].squeeze().cpu())

            last_state['rot'] = state['rot'][..., -1, :].squeeze()
            if not motion_mode:
                last_state['pos'] = state['pos'][..., -1, :].squeeze()
                last_state['vel'] = state['vel'][..., -1, :].squeeze()
                last_state['cov'] = state['cov'].to(device)

        poses = torch.stack(poses, axis=0)
        vels  = torch.stack(vels,  axis=0)
        covs  = torch.stack(covs,  axis=0)

        rots = pp.SO3(torch.stack([
            r.tensor() if isinstance(r, pp.LieTensor) else r for r in rots
        ], axis=0))

        output_state = {
            'pos': poses.to(self.device),
            'rot': rots.to(self.device),
            'vel': vels.to(self.device),
            'cov': covs.to(self.device)
        }
        return output_state
