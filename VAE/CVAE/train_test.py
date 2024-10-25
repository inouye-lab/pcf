import torch
import torch.optim as optim
import time
from tqdm import trange
import numpy as np
import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from torch import nn
import wandb 

def train(model, train_loader, valid_loader, args):
    # based on https://github.com/osu-srml/CF_Representation_Learning/blob/master/CVAE/train.py
    device = args.device
    model.to(device)
    model = model.train()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # ------------- Log ------------- #
    train_x_recon_losses = []
    train_y_recon_losses = []
    train_u_kl_losses = []
    valid_x_recon_losses = []
    valid_y_recon_losses = []
    valid_u_kl_losses = []

    loss_val_log = []
    epoch_log = []
    
    best_epoch = 0
    best_loss = 1e10
    start_time = time.time()
    # ------------- Log ------------- #

    for epoch_i in trange(args.n_epochs):
        model.train()
        loss_whole = 0
        for idx, (r, d, a, y) in enumerate(train_loader):
            loss, x_recon_loss, y_recon_loss, u_kl_loss = model.calculate_loss(r.to(device), d.to(device), a.to(device), y.to(device)) 

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_x_recon_losses.append(x_recon_loss.item())
            train_y_recon_losses.append(y_recon_loss.item())
            train_u_kl_losses.append(u_kl_loss.item())
            loss_whole += loss.cpu().detach().numpy()

        epoch_log.append(epoch_i) 

        loss_reconx = np.array(train_x_recon_losses[-len(train_loader):]).mean()
        loss_kl = np.array(train_u_kl_losses[-len(train_loader):]).mean()
        loss_recony = np.array(train_y_recon_losses[-len(train_loader):]).mean()

        if args.wandb:
            wandb.log({'Loss': loss_whole},step=epoch_i)
            wandb.log({'BCE(x)': loss_reconx},step=epoch_i)
            wandb.log({'KL(u)' : loss_kl},step=epoch_i)
            wandb.log({'BCE(y)' : loss_recony},step=epoch_i)


        model.eval()
        loss_whole = 0
        _all = 0
        with torch.no_grad():
            for idx, (r, d, a, y) in enumerate(valid_loader):
                loss_val, x_recon_loss_val, y_recon_loss_val, u_kl_loss_val = model.calculate_loss(r.to(device), d.to(device), a.to(device), y.to(device))  # (*cur_batch)

                valid_x_recon_losses.append(x_recon_loss_val.item())
                valid_y_recon_losses.append(y_recon_loss_val.item())
                valid_u_kl_losses.append(u_kl_loss_val.item())
                loss_whole += loss_val.cpu().detach().numpy()
                _all += float(y.size(0))
            
            loss_val_log.append(loss_whole)
            loss_check = loss_whole.item() / _all

            # if epoch_i == 0 and loss_check > best_loss:
            #     best_loss = loss_check

            print('now best epoch is, best loss, loss_check', best_epoch, best_loss, loss_check)
            print('loss_check < best_loss', loss_check < best_loss)

            if loss_check < best_loss:
                #model_path = os.path.join(args.save_path, 'model.pth')
                torch.save(model.state_dict(), args.save_path / 'model.pth')
                best_epoch = epoch_i
                best_loss = loss_check
                print('best epoch update by loss, epoch is ', epoch_i)

            if epoch_i - best_epoch > args.break_epoch and args.early_stop == True:
                line = 'time elapsed: {:.4f}min'.format((time.time() - start_time) / 60.0)
                #logger.info(line)
                break

        if args.early_stop == False:
            torch.save(model.state_dict(), args.save_path / 'model.pth')

        line = 'time elapsed: {:.4f}min'.format((time.time() - start_time) / 60.0)
        print(line)
