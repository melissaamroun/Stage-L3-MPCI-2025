#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version idiomatique PyTorch du réseau Keras original.
Format d'entrée attendu : channels-first (N, C=1, H, W).
"""

from typing import Optional
import torch
import torch.nn as nn
from torchsummary import summary
from Parameters import Params


class SquareActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class LogActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(x.dtype).eps if x.is_floating_point() else 1e-7
        return torch.log(x + eps)


class MeanSquarePooling(nn.Module):
    """Define the mean-square pooling block.
    """

    def __init__(self, activation: str = "square"):
        super().__init__()
        if activation == "square":
            self.act = SquareActivation()
        else:
            self.act = None
        self.log = LogActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act is not None:
            x = self.act(x)
        return self.log(x.mean(dim=(2, 3)))


def _increment_filters_tensor(out_ch: int, in_ch: int, device=None, dtype=torch.float32):
    """
    Create a small 3x3 filter tensor reproducing the spirit of IncrementFilters
    used in TF code. The original used shape (3,3,1,nf). We'll create (out_ch, in_ch, 3,3)
    and fill first few output channels with the pattern when possible.
    This is a best-effort; it only affects the first conv if requested.
    """
    w = torch.zeros((out_ch, in_ch, 3, 3), dtype=dtype, device=device)
    # pattern derived from original:
    # filters[0, 0, 0, :] = 1     
    # filters[0, 1, 0, 0] = -2
    # filters[1, 0, 0, 1] = -2
    # filters[0, 2, 0, 0] = 1
    # filters[2, 0, 0, 1] = 1
    # We'll apply a simple pattern to some output channels if size permits.
    for k in range(min(out_ch, 4)):
        if in_ch >= 1:
            # set center to 1 for channel k
            w[k, 0, 1, 1] = 1.0
    if out_ch >= 2 and in_ch >= 1:
        w[0, 0, 1, 0] = -2.0
        w[1, 0, 0, 1] = -2.0
    return w


class ConvPyramid(nn.Module):
    """Definition of a convolutional pyramid.
    """

    def __init__(
        self,
        nlayers: int = 5,
        base_channels: int = 1,
        group: bool = True,
        skip_connection: bool = True,
        sym: bool = True,
        trainable: bool = True,
        use_increment_init: bool = False,
    ):
        """
        Conv pyramid: successive 3x3 convolutions doubling the number
        of channels each layer channels-first format.
        """
        super().__init__()
        self.nlayers = nlayers
        self.skip_connection = skip_connection
        self.sym = sym
        self.use_increment_init = use_increment_init
        self.msp = MeanSquarePooling()  # returns (N, C)

        convs = []
        in_ch = base_channels
        out_ch = base_channels * 2
        for j in range(nlayers):
            if group:
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1,
                                 groups=in_ch, padding=1, bias=False)
            else:
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1,
                                 padding=1, bias=False)
            convs.append(conv)
            in_ch = out_ch
            out_ch = out_ch * 2
        self.convs = nn.ModuleList(convs)  # (N, C, 1, 1) or (N, C)

        # freeze conv weights if trainable==False
        if not trainable:
            for p in self.parameters():
                p.requires_grad = False

    def reset_parameters_with_increment(self, device=None, dtype=None):
        """Optional: initialize first conv with Increment-like pattern."""
        if len(self.convs) == 0:
            return
        first = self.convs[0]
        with torch.no_grad():
            if dtype is None:
                dtype = first.weight.dtype
            if device is None:
                device = first.weight.device
            out_ch, in_ch, kh, kw = first.weight.shape
            inc = _increment_filters_tensor(out_ch, in_ch, device=device,
                                            dtype=dtype)
            # if shapes match, copy, else skip
            if inc.shape == first.weight.shape:
                first.weight.copy_(inc)
            # for other convs keep default initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        outputs = []
        y = x
        for conv in self.convs:
            y = conv(y)  # (N, out_ch, H, W)
            if self.skip_connection:
                outputs.append(self.msp(y))
                if self.sym:
                    outputs.append(self.msp(y.permute(0, 1, 3, 2)))

        if not self.skip_connection:
            outputs = [self.msp(y)]
            if self.sym:
                outputs.append(self.msp(y.permute(0, 1, 3, 2)))

        return torch.cat(outputs, dim=1)
    
    
    
class ConvFlat(nn.Module):
    """Definition of a convolutional pyramid.
    """

    def __init__(
        self,
        nlayers: int = 5,
        base_channels: int = 1,
        group: bool = True,
        skip_connection: bool = True,
        sym: bool = True,
        trainable: bool = True,
        use_increment_init: bool = False,
    ):
        """
        Conv pyramid: successive 3x3 convolutions doubling the number
        of channels each layer channels-first format.
        """
        super().__init__()
        self.nlayers = nlayers
        self.skip_connection = skip_connection
        self.sym = sym
        self.use_increment_init = use_increment_init
        self.msp = MeanSquarePooling()  # returns (N, C)

        convs = []
        in_ch = base_channels
        out_ch = base_channels * 2
        for j in range(nlayers):
            if group:
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=3*j, stride=1,
                                 groups=in_ch, padding=1, bias=False)
            else:
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=3*j, stride=1,
                                 padding=1, bias=False)
            convs.append(conv)
            in_ch = out_ch
            out_ch = out_ch * 2
        self.convs = nn.ModuleList(convs)  # (N, C, 1, 1) or (N, C)

        # freeze conv weights if trainable==False
        if not trainable:
            for p in self.parameters():
                p.requires_grad = False



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        outputs = []
        for conv in self.convs:
            y = conv(x)  # (N, out_ch, H, W)
            if self.skip_connection:
                outputs.append(self.msp(y))
                if self.sym:
                    y = conv(x.permute(0, 1, 3, 2))
                    outputs.append(self.msp(y))

        if not self.skip_connection:
            outputs = [self.msp(y)]
            if self.sym:
                outputs.append(self.msp(y.permute(0, 1, 3, 2)))

        return torch.cat(outputs, dim=1)


def build_mlp(input_dim: int, nlayers: int, hidden_units: int,
              activation: Optional[str], activation_last: Optional[str],
              out_dim: int):
    """Define the dense layers of the top MLP.
    """
    layers = []
    in_dim = input_dim
    act_layer = None
    if activation == "relu":
        act_layer = nn.ReLU
    elif activation == "tanh":
        act_layer = nn.Tanh
    elif activation == "sigmoid":
        act_layer = nn.Sigmoid
    if activation_last == "sigmoid":
        act_layer_last = nn.Sigmoid
    else:
        act_layer_last = None

    for i in range(nlayers):
        layers.append(nn.Linear(in_dim, hidden_units))
        if act_layer is not None:
            layers.append(act_layer())
        in_dim = hidden_units

    # final layer
    layers.append(nn.Linear(in_dim, out_dim))
    if act_layer_last is not None:
        layers.append(act_layer_last())
    return nn.Sequential(*layers)


class CNNModel(nn.Module):
    """Definition of the complete CNN architecture.
    """

    def __init__(self, params, use_increment_init: bool = False):
        """
        params should be an object (or namespace) with attributes:
          - nlayers_conv (int)
          - skip_connections (bool)
          - train_conv (bool)
          - activation_conv (str)  # "square" or "squaremax"
          - nlayers_dense (int)
          - units_dense (int)
          - activation_dense (str)  # "relu"/"tanh"/None
          - activation_last (str)  # "sigmoid"/None
          - cells_last (int)
          - activation_last (str) or None
          - sym (bool)
        """
        super().__init__()
        self.params = params

        # conv pyramid: start with 1 input channel
        #if params['model']['archi'] == "CNN pyramid":
        self.cp = ConvPyramid(
                nlayers=params['nlayers_conv'],
                base_channels=1,
                group=params['group_conv'],
                skip_connection=params['skip_connections'],
                sym=params['sym'],
                trainable=params['train_conv'],
                use_increment_init=use_increment_init
            )
        #elif params['model']['archi'] == "CNN flat" :
            #self.cp = ConvFlat(
             #   nlayers=params['nlayers_conv'],
              #  base_channels=1,
               # group=params['group_conv'],
                #skip_connection=params['skip_connections'],
                #sym=params['sym'],
                #trainable=params['train_conv'],
                #use_increment_init=use_increment_init
            #)
                
        if use_increment_init:
            self.cp.reset_parameters_with_increment()
        


        # We'll set up MLP lazily once we know the pooled channel dimension
        self._mlp = None
        self._built = False

    def _build_head(self, pooled_channels: int):
        """
        Create MLP head. If sym, pooled channels doubled.
        """
        in_dim = pooled_channels
        self._mlp = build_mlp(in_dim, self.params['nlayers_dense'],
                              self.params['units_dense'],
                              self.params['activation_dense'],
                              self.params['activation_last'],
                              self.params['cells_last'])
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C=1, H, W)
        returns: (N, out_dim)  - flattened final outputs (idomatic)
        """
        # conv pyramid -> (N, C_total, H, W)
        y = self.cp(x)

        if not self._built:
            self._build_head(y.shape[1])

        out = self._mlp(y)  # (N, cells_last)
        return out


def CreateModel(params, device: Optional[torch.device] = None,
                use_increment_init: bool = False,
                verbose=True):
    """Model creation.
    """
    
    model = CNNModel(params, use_increment_init=use_increment_init)
    
    if device is not None:
        model.to(device)
    # dummy forward to initialize lazy head
    H, W = 128, 128  # expects tuple (H, W)
    dummy = torch.randn(1, 1, H, W, device=device)
    _ = model(dummy)
    if verbose:
        summary(model, (1, H, W))
    return model


if __name__ == '__main__':
    params = Params()
    model = CreateModel(params)
