import numpy as np
from sklearn.metrics import mean_squared_error
import torch

def infer_u(model, r, d, a):
    device = r.device
    
    u_mu, u_logvar = model.q_u(r.to(device), d.to(device), a.to(device))

    u_prev = model.reparameterize(u_mu, u_logvar)
    return u_prev

def gen_x(model, u, a):
    device = u.device

    r, d, _ = model.reconstruct_hard(u, a.to(device))
    x = torch.cat([r, d], dim=1)

    return x

def cf_eval(y, y_cf, a):
    a = a.squeeze()
    mask1 = (a == 0)
    mask2 = (a == 1)
    
    cf_effect = np.abs(y_cf - y)
    o1 = cf_effect[mask1]
    o2 = cf_effect[mask2]
    return np.sum(cf_effect) / cf_effect.shape[0], np.sum(o1) / o1.shape[0], np.sum(o2) / o2.shape[0]


def pcf_mix(y_score, ycf_score, a, is_cf=False):
    # attribute corresponding to y
    a_0_indices = a == 0
    a_1_indices = a == 1
    a_0_ratio = np.sum(a_0_indices) / len(a)
    a_1_ratio = 1-a_0_ratio
    if is_cf is True:
        # we need to use the ratio in the real data
        a_0_ratio, a_1_ratio = a_1_ratio, a_0_ratio

    y_output = np.zeros_like(y_score.ravel())
    y_output[a_0_indices] = y_score[a_0_indices] * a_0_ratio + ycf_score[a_0_indices] * a_1_ratio
    y_output[a_1_indices] = y_score[a_1_indices] * a_1_ratio + ycf_score[a_1_indices] * a_0_ratio

    return y_output
