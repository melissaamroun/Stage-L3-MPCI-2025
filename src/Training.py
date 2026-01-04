#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 11:22:37 2025

@author: frichard
"""
import yaml


import os
#import torch
#import numpy as np
from PrepareDatasets import CreateDatasets
from architecture import CreateModel
from torch import save, load
from numpy import inf
from torch import device as torch_device
from torch.cuda import is_available as cuda_is_available
from torch.optim import Adam
from torch.nn import MSELoss
#from matplotlib import pyplot as plt
from tqdm import tqdm


rep_expe = "./Experiments/Experiments-011/"

with open(rep_expe + "setting.yaml", 'r') as stream:
    params = yaml.safe_load(stream)
print(params)


# Model creation.
model = CreateModel(params['model'])


# Preparation of optimization.
device = torch_device("cuda" if cuda_is_available() else "cpu")
optimizer = Adam(model.parameters(), params['optimization']['lr'])
lossfunction = MSELoss(reduction='mean')
nb_epochs = params['optimization']['nb_epochs']

dataset, loader = CreateDatasets(params)

epoch0 = 0
rep_out = rep_expe + "checkpoints"
if not os.path.isdir(rep_out):
    os.mkdir(rep_out)
elif os.path.isfile(rep_out + "/model.pt"):
    state_dict = load(rep_out + "/model.pt")
    model.load_state_dict(state_dict['model_state'], strict=True)
    model = model.to(device)
    model.eval()
    epoch0 = state_dict['epoch']
    print(f"Starting at epoch {epoch0} with existing model.")

best_loss_val = inf
for epoch in range(epoch0, nb_epochs):
    for phase in ['Training', 'Validation']:
        running_loss = 0.0
        #for i, (data_in, data_out) in enumerate(loader[phase]):
        for i, (data_in, data_out) in enumerate(tqdm(loader[phase], desc=f"{phase} epoch {epoch}")):

            data_in = data_in.to(device)
            data_out = data_out.to(device)
            if phase == 'Training':
                optimizer.zero_grad()
            data_out_est = model(data_in)
            loss = lossfunction(data_out, data_out_est)
            if phase == 'Training':
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
        running_loss /= len(loader[phase])

        if phase == 'Training':
            train_loss = running_loss
        else:
            val_loss = running_loss


    print(f"Epoch {epoch}: train loss = {train_loss:.6f}"
          f"-- val loss = {val_loss:.6f}")

    if val_loss < best_loss_val:
        best_loss_val = val_loss
        save({"epoch": epoch,
              "model_state": model.state_dict(),
              "optimizer_state": optimizer.state_dict()
              }, rep_out + "/model.pt")


"""Ajouts au code"""

"""epoch0 = 0
rep_out = rep_expe + "checkpoints"
if not os.path.isdir(rep_out):
    os.mkdir(rep_out)
elif os.path.isfile(rep_out + "/model.pt"):
    state_dict = load(rep_out + "/model.pt")
    model.load_state_dict(state_dict['model_state'], strict=True)
    model = model.to(device)
    model.eval()
    epoch0 = state_dict['epoch']
    print(f"Starting at epoch {epoch0} with existing model.")


phase = 'Test'

running_loss = 0.0 # À revoir
model.eval()

H_true = []
H_pred = [] #enumerate(tqdm(loader[phase], desc=f"{phase} epoch {epoch}"))

with torch.no_grad():
    for i, (data_in, data_out) in enumerate(tqdm(loader[phase], desc=f"{phase} epoch {epoch}")):
        data_in = data_in.to(device)
        data_out = data_out.to(device)

        data_out_est = model(data_in)

        H_true.append(data_out.squeeze(1).cpu().numpy())
        H_pred.append(data_out_est.squeeze(1).cpu().numpy())

        loss = lossfunction(data_out, data_out_est)
        running_loss += loss.item()

running_loss /= len(loader[phase])
test_loss = running_loss

H_true = np.concatenate(H_true)
H_pred = np.concatenate(H_pred)

biais = np.mean(H_pred - H_true)
RMSE = np.sqrt(np.mean((H_pred - H_true) ** 2))

print(f"Test loss = {test_loss:.6f}")
print(H_true)
print(H_pred)
#print("RMSE loss :", RMSE)
print("Biais :", biais)



plt.plot(H_true, H_pred, 'k+', label="Model predictions")
plt.plot(H_true, H_true, label="Ideal: y = x")
plt.xlabel('True H')
plt.ylabel('Estimated H')
plt.title('Estimation of the Hurst index')
plt.legend()
plt.grid(True)
plt.show()"""



"""running_loss = 0.0
model.eval() 

H_true = []
H_pred = []

with torch.no_grad():
    for data_in, data_out in loader[phase]:
        data_in = data_in.to(device)
        data_out = data_out.to(device)
        
        prediction = model(data_in)
        
        H_true.append(data_out.squeeze(1).cpu().numpy())
        H_pred.append(prediction.squeeze(1).cpu().numpy())
    
H_true = np.concatenate(H_true)
H_pred = np.concatenate(H_pred)

print(H_true)
print(H_pred)


plt.plot(H_true, H_true)
plt.plot(H_true, H_pred, 'k+')
#plt.show()

biais = np.mean(H_pred - H_true)
RMSE = np.sqrt(np.mean((H_pred - H_true) ** 2))"""
        
        
        
        
        
        
        
        
        
        
        
        
