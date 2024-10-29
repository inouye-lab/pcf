import argparse

import numpy as np
import pandas as pd
import torch
import os
import logging.handlers
from pathlib import Path

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=256, help='number of gpu')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate') #(1e-3)/2
parser.add_argument('--loss_fn', type=str, default='BCE', help='loss function')
parser.add_argument('--break_epoch', type=int, default=30, help='break epoch')
parser.add_argument('--act_fn', type=str, default='Tanh', help='activation function')

parser.add_argument('--model',type=str, default='cvae', help='model')
parser.add_argument('--a_y', type=float, default=1,  help='hyper-parameter for y')
parser.add_argument('--a_r', type=float, default=1, help='hyper-parameter for x_r')
parser.add_argument('--a_d', type=float, default=1, help='hyper-parameter for x_d')
parser.add_argument('--a_a', type=float, default=1, help='hyper-parameter for a recon from z')
parser.add_argument('--a_f', type=float, default=0, help='hyper-parameter for fairness')
parser.add_argument('--a_h', type=float, default=0.4, help='hyper-parameter for independency')
parser.add_argument('--u_kl',  type=float, default=1, help='hyper-parameter for u_kl')
parser.add_argument('--ur_dim', type=int, default=3, help='dim of ur')
parser.add_argument('--ud_dim', type=int, default=4, help='dim of ud')
parser.add_argument('--h_dim', type=int, default=100, help='dim of ud')

parser.add_argument('--run', type=int, default=2, help='# of run')
parser.add_argument('--gpu', type=int, default=0, help='number of gpu')
parser.add_argument('--rep', type=int, default=0, help='number of rep')

parser.add_argument('--use_label', default=False, action='store_true')
parser.add_argument("--use_real", type=bool, default=False, help="Use real dat or not")
parser.add_argument("--normalize", action='store_true', default=False, help="normalize or not")
parser.add_argument("--path", type=bool, default=False, help="True/False")
parser.add_argument("--path_attribute", type=str, default="GPA", help="which atrribute is ignored")

parser.add_argument('--retrain', type=bool, default=False, help='True/False')
parser.add_argument('--debug', type=bool, default=True, help='True/False')
parser.add_argument('--test', type=bool, default=True, help='True/False')
parser.add_argument('--tSNE', type=bool, default=True, help='True/False')
parser.add_argument('--clf', type=bool, default=True, help='True/False')
parser.add_argument('--balance', type=bool, default=False, help='True/False')
parser.add_argument('--early_stop', type=bool, default=True, help='True/False')

parser.add_argument('--dataset', type=str, default='adult', help='adult or law')

parser.add_argument('--no_wandb', action='store_true', help='whether to use wandb')
parser.add_argument('--project_name', default='NeurIPS24-CFF-Test')
parser.add_argument('--wandb_entity', default='')
parser.add_argument('--sweep', action='store_true', default=False)
parser.add_argument('--run_name', default='run')
parser.add_argument('--global_path', default='saved')

args = parser.parse_args()



def main(args):
    args.seed = args.run
    args.wandb = not args.no_wandb

    '''GPU setting'''

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    '''Save path setting & mkdir'''
    src_path = Path(os.path.dirname(os.path.realpath('__file__')))

    result_path = src_path / f"{args.global_path}/{args.dataset}/gt/{args.model}"


    if args.model == 'dcevae':
        from model import DCEVAE    
        from DCEVAE.train_test import train
        from DCEVAE.utils import make_adult_loader, make_balancing_loader, make_law_loader
    else:
        raise ValueError("Model not supported")

    '''Load Dataset'''
    print("Load Dataset")
    if args.dataset == "law":
        data_df = pd.read_csv(os.path.join(src_path, "../data/raw/law/law_data.csv"))
        train_loader, valid_loader, test_loader, input_dim = make_law_loader(data_df, args)
        print(input_dim)
    else:
        train_df = open(os.path.join(src_path, '../data/raw/adult/list_attr_adult.txt'))
        if args.balance == True:
            train_loader, valid_loader, test_loader, input_dim = make_balancing_loader(train_df, args)
        else:
            train_loader, valid_loader, test_loader, input_dim = make_adult_loader(train_df, args)
        print(input_dim)
    args.input_dim = input_dim

    if args.model == 'dcevae':
        args.save_path = result_path / 'a_r_{:s}_a_d_{:s}_a_y_{:s}_a_h_{:s}_a_f_{:s}_u_{:s}_ur_{:d}_ud_{:d}_run_{:d}_use_label_{:s}'\
                                  .format(str(args.a_r), 
                                          str(args.a_d), 
                                          str(args.a_y), 
                                          str(args.a_h), 
                                          str(args.a_f),
                                          str(args.u_kl), 
                                          args.ur_dim, 
                                          args.ud_dim, 
                                          args.run, 
                                          str(args.use_label))
        model = DCEVAE(r_dim=input_dim['r'], 
                       d_dim=input_dim['d'], 
                       sens_dim=input_dim['a'], 
                       label_dim=input_dim['y'],
                       args=args).to(args.device)
    else:
        raise ValueError("Model not supported")

    args.save_path.mkdir(parents=True, exist_ok=True)

    if args.wandb or args.sweep:
        wandb.init(project=args.project_name if not args.sweep else None,
                   entity=args.wandb_entity if not args.sweep else None,
                   name=args.run_name,
                   config=vars(args))
        wandb.run.log_code()

    print("Training")
    train(model, train_loader, valid_loader, args)
    #test(test_loader, args)

    # ----------------- Generate Simulated Dataset ----------------- # 
    def infer_u(model, r, d, a, y):
        device = r.device
        d = d.to(device)
        a = a.to(device)
        y = y.to(device)
        u_mu, u_logvar = model.q_u(r, d, a, y)
        u_prev = model.reparameterize(u_mu, u_logvar)
        return u_prev

    #model.load_state_dict(torch.load(args.save_path / 'model.pth'))
    model = torch.load(args.save_path / 'model.pth')

    all_splits = ['train', 'valid', 'test']
    data_dict = dict()
    for split in all_splits:
        data_dict[split] = dict()

    for split in all_splits:
        data_dir = Path(f'../data/processed/{args.dataset}/{args.seed}') / f"{split}.npz"
        data_real = np.load(data_dir)
        x = torch.Tensor(data_real['input_real'])

        if args.dataset == 'adult':
            r,d = torch.Tensor(x[:, :3]), torch.Tensor(x[:, 3:])
            assert r.shape[1] == 3, 'check dim of r'
            assert d.shape[1] == 6, 'check dim of d'
        a = torch.Tensor(data_real['a'])
        #a = torch.distributions.Bernoulli(0.5).sample([len(a)]).unsqueeze(1)
        y = torch.Tensor(data_real['y_real'])
        r = r.to(args.device)
        a = a.to(args.device)
        u_post = infer_u(model, r, d, a, y)
        u = torch.randn_like(u_post)

        r,d, _, y ,_ = model.reconstruct_hard(u, a)

        x = torch.cat((r,d), 1)

        data_dict[split]['x'] = x.detach().cpu()
        data_dict[split]['u'] = u.detach().cpu()
        data_dict[split]['a'] = a.detach().cpu()
        data_dict[split]['y'] = y.detach().cpu()

        # generate cf
        a_cf = 1-a
        r_cf,d_cf, _, y_cf, _ = model.reconstruct_hard(u, a_cf)
        x_cf = torch.cat((r_cf,d_cf), 1)
        data_dict[split]['x_cf'] = x_cf.detach().cpu()
        data_dict[split]['a_cf'] = a_cf.detach().cpu()
        data_dict[split]['y_cf'] = y_cf.detach().cpu()

    torch.save(data_dict, args.save_path / "data_dict.pth")

main(args)