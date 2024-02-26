#!/usr/bin/env python
# coding: utf-8
"""
Train with different method
"""

import time

import torch
from torch import nn
from tqdm import tqdm

import pyepo
import net
from earlystop import earlyStopper
from gradnorm import gradNorm

class LinearRegression(nn.Module):
    """
    Linear layer model with softplus
    """
    def __init__(self, p, m):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(p, m)
        self.softp = nn.Softplus(threshold=5)

    def forward(self, x):
        h = self.linear(x)
        out = self.softp(h)
        return out

def our_train2S(dataloader, dataloader_val, optmodels, data_params, train_params):
    """
    Train two-stage model with MSE
    """
    m = data_params["item"] # number of nodes
    p = data_params["feat"] # size of feature
    num_epochs = train_params["epoch"] # number of epochs
    val_step = train_params["val_step"] # validation steps
    lr = train_params["lr"] # learnin rate
    loss_log = []
    # init model
    reg = LinearRegression(p, m) # TODO subject to change
    # cuda
    if torch.cuda.is_available():
        reg = reg.cuda()
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # set loss
    loss_func = nn.MSELoss()
    reg.train()

    tbar = tqdm(range(num_epochs))
    for epoch in tbar:
        for i, data in enumerate(dataloader):
            x, c, w, z = data
            if torch.cuda.is_available():
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # forward pass
            cp = reg(x)

            loss = loss_func(cp, c)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            loss_log.append(loss.item())
            tbar.set_description("Loss: {:3.4f}".format(loss.item()))
    return reg, loss_log, 0, 0, None

def our_trainSeperatedMSE(dataloader, dataloader_val, optmodels, data_params, train_params):
    """
    Train with separated model
    """
    m = data_params["item"] # number of nodes
    p = data_params["feat"] # size of feature
    num_epochs = train_params["epoch"] # number of epochs
    val_step = train_params["val_step"] # validation steps
    lr = train_params["lr"] # learnin rate
    proc = train_params["proc"] # process number for optmodel
    # init
    reg = {}
    loss_log = {}
    elapsed, elapsed_val = 0, 0
    # per task
    print("Cuda available: ", torch.cuda.is_available())
    for ind, (task, optmodel) in enumerate(optmodels.items()):
        reg[task] = LinearRegression(p, m) 
        # cuda
        if torch.cuda.is_available():
            reg[task] = reg[task].cuda()
        # set optimizer
        optimizer = torch.optim.Adam(reg[task].parameters(), lr=lr)
        # set loss
        loss_func = pyepo.func.SPOPlus(optmodel, processes=proc)
        # set MSE loss
        MSE = nn.MSELoss()
        # set stopper
        stopper = earlyStopper(patience=5)
        # train mode
        reg[task].train()
        # init log
        loss_log[task] = []
        # start traning
        tbar = tqdm(range(num_epochs))
        for epoch in tbar:
            tick = time.time()
            # load data
            for data in dataloader:
                x, c, w, z = data
                # cuda
                if torch.cuda.is_available():
                    x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
                # forward pass
                cp = reg[task](x)
                loss = loss_func(cp, c, w[:,ind], z[:,ind]).mean() + MSE(cp, c)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # log
                loss_log[task].append(loss.item())
                tbar.set_description("Loss: {:3.4f}".format(loss.item()))
            # time
            tock = time.time()
            elapsed += tock - tick
            # early stop
            if epoch % val_step == 0:
                tick = time.time()
                loss = 0
                with torch.no_grad():
                    for data in dataloader_val:
                        x, c, w, z = data
                        # cuda
                        if torch.cuda.is_available():
                            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
                        # forward pass
                        cp = reg[task](x)
                        loss += loss_func(cp, c, w[:,ind], z[:,ind]).mean() + MSE(cp, c)
                tock = time.time()
                elapsed_val += tock - tick
                if stopper.stop(loss):
                    break
    return reg, loss_log, elapsed, elapsed_val, None


def trainComb(dataloader, dataloader_val, optmodels, data_params, train_params):
    """
    Train with simple combination
    """
    m = data_params["item"] # number of nodes
    p = data_params["feat"] # size of feature
    num_epochs = train_params["epoch"] # number of epochs
    val_step = train_params["val_step"] # validation steps
    lr = train_params["lr"] # learnin rate
    proc = train_params["proc"] # process number for optmodel
    # init model
    reg = LinearRegression(p, m) 
    mtl = net.mtlSPO(reg, optmodels.values(), processes=proc)
    # cuda
    if torch.cuda.is_available():
        reg = reg.cuda()
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # set stopper
    stopper = earlyStopper(patience=5)
    # train mode
    mtl.train()
    # init log
    loss_log = []
    # start traning
    tbar = tqdm(range(num_epochs))
    elapsed, elapsed_val = 0, 0
    for epoch in tbar:
        tick = time.time()
        # load data
        for data in dataloader:
            # cuda
            if torch.cuda.is_available():
                data = [d.cuda() for d in data]
            # forward pass
            loss = mtl(*data).sum()
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            loss_log.append(loss.item())
            tbar.set_description("Loss: {:3.4f}".format(loss.item()))
        # time
        tock = time.time()
        elapsed += tock - tick
        # early stop
        if epoch % val_step == 0:
            tick = time.time()
            loss = 0
            with torch.no_grad():
                for data in dataloader_val:
                    # cuda
                    if torch.cuda.is_available():
                        data = [d.cuda() for d in data]
                    # forward pass
                    loss += mtl(*data).sum()
            tock = time.time()
            elapsed_val += tock - tick
            if stopper.stop(loss):
                break
    return reg, loss_log, elapsed, elapsed_val, None


def trainCombMSE(dataloader, dataloader_val, optmodels, data_params, train_params):
    """
    Train with simple combination and MSE
    """
    m = data_params["item"] # number of nodes
    p = data_params["feat"] # size of feature
    num_epochs = train_params["epoch"] # number of epochs
    val_step = train_params["val_step"] # validation steps
    lr = train_params["lr"] # learnin rate
    proc = train_params["proc"] # process number for optmodel
    # init model
    reg = LinearRegression(p, m) 
    mtl = net.mtlSPO(reg, optmodels.values(), processes=proc, mse=True)
    # cuda
    if torch.cuda.is_available():
        reg = reg.cuda()
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # set stopper
    stopper = earlyStopper(patience=5)
    # train mode
    mtl.train()
    # init log
    loss_log = []
    # start traning
    tbar = tqdm(range(num_epochs))
    elapsed, elapsed_val = 0, 0
    for epoch in tbar:
        tick = time.time()
        # load data
        for data in dataloader:
            # cuda
            if torch.cuda.is_available():
                data = [d.cuda() for d in data]
            # forward pass
            loss = mtl(*data).sum()
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            loss_log.append(loss.item())
            tbar.set_description("Loss: {:3.4f}".format(loss.item()))
        # time
        tock = time.time()
        elapsed += tock - tick
        # early stop
        if epoch % val_step == 0:
            tick = time.time()
            loss = 0
            with torch.no_grad():
                for data in dataloader_val:
                    # cuda
                    if torch.cuda.is_available():
                        data = [d.cuda() for d in data]
                    # forward pass
                    loss += mtl(*data).sum()
            tock = time.time()
            elapsed_val += tock - tick
            if stopper.stop(loss):
                break
    return reg, loss_log, elapsed, elapsed_val, None


def trainGradNorm(dataloader, dataloader_val, optmodels, data_params, train_params):
    """
    Train with GradNorm
    """
    m = data_params["item"] # number of nodes
    p = data_params["feat"] # size of feature
    num_epochs = train_params["epoch"] # number of epochs
    val_step = train_params["val_step"] # validation steps
    lr = train_params["lr"] # learnin rate
    lr2 = train_params["lr2"] # learnin rate for weights
    proc = train_params["proc"] # process number for optmodel
    alpha = train_params["alpha"] # hyperparameter of restoring force
    # init model
    reg = LinearRegression(p, m)
    mtl = net.mtlSPO(reg, optmodels.values(), processes=proc)
    # cuda
    if torch.cuda.is_available():
        reg = reg.cuda()
    # train mode
    mtl.train()
    # start traning
    weights_log, loss_log, elapsed, elapsed_val = gradNorm(net=mtl, layer=reg.linear,
                                                           alpha=alpha, dataloader=dataloader,
                                                           dataloader_val=dataloader_val,
                                                           num_epochs=num_epochs, lr1=lr,
                                                           lr2=5e-3, val_step=val_step)
    return reg, loss_log, elapsed, elapsed_val, weights_log


def trainGradNormMSE(dataloader, dataloader_val, optmodels, data_params, train_params):
    """
    Train with GradNorm and MSE
    """
    m = data_params["item"] # number of nodes
    p = data_params["feat"] # size of feature
    num_epochs = train_params["epoch"] # number of epochs
    val_step = train_params["val_step"] # validation steps
    lr = train_params["lr"] # learnin rate
    lr2 = train_params["lr2"] # learnin rate for weights
    proc = train_params["proc"] # process number for optmodel
    alpha = train_params["alpha"] # hyperparameter of restoring force
    # init model
    reg = LinearRegression(p, m)
    mtl = net.mtlSPO(reg, optmodels.values(), processes=proc, mse=True)
    # cuda
    if torch.cuda.is_available():
        reg = reg.cuda()
    # train mode
    mtl.train()
    # start traning
    weights_log, loss_log, elapsed, elapsed_val = gradNorm(net=mtl, layer=reg.linear,
                                                           alpha=alpha, dataloader=dataloader,
                                                           dataloader_val=dataloader_val,
                                                           num_epochs=num_epochs, lr1=lr,
                                                           lr2=5e-3, val_step=val_step)
    return reg, loss_log, elapsed, elapsed_val, weights_log
