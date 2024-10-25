import torch
from torch import nn
import torch.distributions as dists
import torch.nn.functional as F
import math

import random
    
class CVAE(nn.Module):
    # based on https://github.com/osu-srml/CF_Representation_Learning/blob/master/CVAE/model.py
    # currently only coded for law dataset 
    def __init__(self, r_dim, d_dim, sens_dim, label_dim, args):
        super(CVAE, self).__init__()
        '''random seed'''
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


        """model structure"""
        self.device = args.device
        self.args = args
        self.r_dim = r_dim
        self.d_dim = d_dim
        self.label_dim = label_dim
        self.sens_dim = sens_dim
        u_dim = args.u_dim
        self.u_dim = u_dim
        
        i_dim = r_dim + d_dim + sens_dim

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(i_dim, i_dim),
            nn.Tanh(),
        )

        self.encoder_mu = nn.Sequential(
            nn.Linear(i_dim, u_dim),
        )

        self.encoder_logvar = nn.Sequential(
            nn.Linear(i_dim, u_dim),
        )

        # decoder
        self.decoder_ua_to_r = nn.Sequential(
            nn.Linear(u_dim, u_dim),
            nn.Tanh(),
            nn.Linear(u_dim, r_dim),
        )

        self.decoder_ua_to_d = nn.Sequential(
            nn.Linear(u_dim + sens_dim, u_dim),
            nn.Tanh(),
            nn.Linear(u_dim, d_dim),
        )

        self.decoder_x_to_y = nn.Sequential(
            nn.Linear(u_dim + r_dim + d_dim, u_dim),
            nn.Tanh(),
            nn.Linear(u_dim, label_dim))

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def rearrange(self, prev, index):
        new = torch.ones_like(prev)
        new[index, :] = prev
        return new

    @staticmethod
    def reparameterize(mu, logvar):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std).to(device)
        return eps.mul(std).add_(mu)

    def diagonal(self, M):
        """
        If diagonal value is close to 0, it could makes cholesky decomposition error.
        To prevent this, I add some diagonal value which won't affect the original value too much.
        """
        new_M = torch.where(torch.abs(M) < 1e-05, M + 1e-05 * torch.abs(M), M)
        return new_M
    
    def q_u(self, r, d, a):
        
        i = torch.cat((r, d, a), 1)

        # q(z|r,d)
        intermediate = self.encoder(i)
        u_mu = self.encoder_mu(intermediate)
        u_logvar = self.encoder_logvar(intermediate)

        return u_mu, u_logvar

    def p_i(self, u, a):
        
        r = self.decoder_ua_to_r(u)

        d = self.decoder_ua_to_d(torch.cat([u, a], 1))

        ux = torch.cat((u, r, d), dim=1)

        y = self.decoder_x_to_y(ux)
        
        return r, d, y

    def reconstruct_hard(self, u, a, add_a_impact=False):
        
        r = self.decoder_ua_to_r(u)
        r_hard = torch.nn.functional.gumbel_softmax(r, tau=1, hard=True)

        d = self.decoder_ua_to_d(torch.cat([u, a], 1))
        if add_a_impact:
            # this is applied only when generating semi-synthetic dataset to amplify the impact of A 
            d = d + 2*a
        d_dist = dists.MultivariateNormal(d, torch.eye(d.size(1)).to(self.device))
        d_hard = d_dist.sample()

        ux = torch.cat((u, r_hard, d_hard), dim=1)
        y = self.decoder_x_to_y(ux)
        y_dist = dists.MultivariateNormal(y, torch.eye(y.size(1)).to(self.device))
        y_hard = y_dist.sample()

        return r_hard, d_hard, y_hard

    def calculate_recon_loss(self, r, d, a, y):
        MB = self.args.batch_size

        u_mu, u_logvar = self.q_u(r, d, a)
        u = self.reparameterize(u_mu, u_logvar)
        r_mu, d_mu, y_p = self.p_i(u, a)
        
        r_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        d_loss_fn = nn.MSELoss(reduction="sum")
        y_loss_fn = nn.MSELoss(reduction="sum")
        
        d_recon = d_loss_fn(d_mu, d) / MB
        r_recon = r_loss_fn(r_mu, r) / MB
        recon = self.args.a_d * d_recon + self.args.a_r * r_recon
        y_recon = y_loss_fn(y_p, y) / MB


        return recon, y_recon, u_mu, u_logvar, y_p


    def calculate_loss(self, r, d, a, y):
        MB = self.args.batch_size
        
        recon, y_recon, u_mu, u_logvar, y_p = self.calculate_recon_loss(r, d, a, y)
        
        """KL loss"""
        #Prohibiting cholesky error
        u_logvar = self.diagonal(u_logvar)

        assert (torch.sum(torch.isnan(u_logvar)) == 0), 'u_logvar'

        u_dist = dists.MultivariateNormal(u_mu.flatten(), torch.diag(u_logvar.flatten().exp()))
        u_prior = dists.MultivariateNormal(torch.zeros(self.u_dim * u_mu.size()[0]).to(self.device),\
                                           torch.eye(self.u_dim * u_mu.size()[0]).to(self.device))
        u_kl = dists.kl.kl_divergence(u_dist, u_prior)/MB

        assert (torch.sum(torch.isnan(recon)) == 0), 'x_recon'
        assert (torch.sum(torch.isnan(y_recon)) == 0), 'y_recon'
        assert (torch.sum(torch.isnan(u_kl)) == 0), 'u_kl'
        
        ELBO = recon + self.args.a_y * y_recon + self.args.u_kl * u_kl

        assert (torch.sum(torch.isnan(ELBO)) == 0), 'ELBO'
        
        return ELBO, recon, y_recon, u_kl


