import argparse
import sys
import pandas as pd
import torch
import os
import logging.handlers
from pathlib import Path


sys.path.append('../')
from adult.dataset import SimUCIAdultTrain

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
parser.add_argument('--dataset_root', default='../VAE/saved/0426')
parser.add_argument('--global_path', default='saved')

args = parser.parse_args()



def main(args):
    args.seed = args.run
    args.wandb = not args.no_wandb

    '''GPU setting'''
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    '''GPU setting'''

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    '''Save path setting & mkdir'''
    src_path = Path(os.path.dirname(os.path.realpath('__file__')))

    result_path = src_path / f"{args.global_path}/{args.dataset}/est/{args.model}"


    if args.model == 'dcevae':
        from model import DCEVAE    
        from DCEVAE.train_test import train
        from DCEVAE.utils import make_adult_loader, make_balancing_loader, make_law_loader
    else:
        raise ValueError("Model not supported")

    if args.dataset == 'adult':
        # always use G(x,y,a,a') for dataset
        train_set = SimUCIAdultTrain(split='train', root = args.dataset_root)
        valid_set = SimUCIAdultTrain(split='valid', root = args.dataset_root)
        test_set = SimUCIAdultTrain(split='test', root = args.dataset_root)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        input_dim = {'r':3, 'd':6, 'a':1, 'y':1}
    else:
        raise NotImplementedError
    
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

main(args)