#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import os
import torch
import numpy as np
from PrepareDatasets import CreateDatasets
from architecture import CreateModel
from torch import load
from torch import device as torch_device
from torch.cuda import is_available as cuda_is_available
from torch.nn import MSELoss
from matplotlib import pyplot as plt
from tqdm import tqdm


rep_expe = "./Experiments/Experiments-011/"

with open(rep_expe + "setting.yaml", 'r') as stream:
    params = yaml.safe_load(stream)
print(params)


device = torch_device("cuda" if cuda_is_available() else "cpu")
lossfunction = MSELoss(reduction='mean')

dataset, loader = CreateDatasets(params)
phase = 'Test'

#Model.
model = CreateModel(params['model']).to(device)

checkpoint_path = rep_expe + "checkpoints/model.pt"
assert os.path.isfile(checkpoint_path), "Checkpoint not found!"

state_dict = load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict['model_state'], strict=True)
model.eval()


#Testing.
running_loss = 0.0
H_true= []
H_pred = []

with torch.no_grad():
    for data_in, data_out in tqdm(loader[phase], desc="Testing"):
        data_in = data_in.to(device)
        data_out = data_out.to(device)

        data_out_est = model(data_in)

        H_true.append(data_out.squeeze(1).cpu().numpy())
        H_pred.append(data_out_est.squeeze(1).cpu().numpy())

        loss = lossfunction(data_out, data_out_est)
        running_loss += loss.item()

test_loss = running_loss / len(loader[phase])

H_true = np.concatenate(H_true)
H_pred = np.concatenate(H_pred)

#Loss.
biais = np.mean(H_pred - H_true)
RMSE = np.sqrt(np.mean((H_pred - H_true) ** 2))

print(f"Test loss = {test_loss:.6f}")
print("Biais =", biais)
print("RMSE =", RMSE)

#Plot creation.
plt.plot(H_true, H_pred, 'k+', label="Model predictions")
plt.plot(H_true, H_true, label="Ideal: y = x")
plt.xlabel('True H')
plt.ylabel('Estimated H')
plt.title('Estimation of the Hurst index')
plt.legend()
plt.grid(True)
plt.show()

