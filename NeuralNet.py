#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 20180215
Update on 20180215
@author: Eduardo Pagotto
'''

#pylint: disable=C0301
#pylint: disable=C0103
#pylint: disable=W0703
#pylint: disable=R0913

import numpy as np

class NeuralNet(object):
    def __init__(self, in_size, out_size, hidden_layers, hidden_size):
        
        w_syn_list = []
        w_bias_list = []

        for index in range(hidden_layers):
            
            #TODO optimizar no append direto
            if index == 0:
                w_syn = 2 * np.random.random((hidden_size, in_size)) - 1
                w_bias = 2 * np.random.random((hidden_size,1)) - 1

            elif index == hidden_layers - 1:
                w_syn = 2 * np.random.random((out_size, hidden_size)) - 1
                w_bias = 2 * np.random.random((hidden_size,1)) - 1
            else:
                w_syn = 2 * np.random.random((hidden_size, hidden_size)) - 1
                w_bias = 2 * np.random.random((out_size,1)) - 1
        
            w_syn_list.append(w_syn)
            w_bias_list.append(w_bias)

    def trainner(self, input, output, epoc_max):
        
        for j in range(epoc_max):
            pass