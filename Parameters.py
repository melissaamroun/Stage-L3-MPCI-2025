#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parameters class.
"""


class Params:
    """Parameter class.
    """

    def __init__(self):
        self.nlayers_conv = 4
        self.skip_connections = True
        self.train_conv = True
        self.activation_conv = "square"
        self.group_conv = True
        self.nlayers_dense = 2
        self.units_dense = 64
        self.activation_dense = "relu"
        self.cells_last = 1
        self.activation_last = "sigmoid"
        self.sym = True
        self.m = (64, 64)
        self.center = False
        self.norm = False
        self.nb_epochs = 10
