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

from ActivactionFunction import sigmoid
from ActivactionFunction import sigmoid_derivative

class NeuralNet(object):
    def __init__(self, in_size, out_size, hidden_layers, hidden_size):
        
        self.w_syn_list = []
        self.w_bias_list = []

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
        
            self.w_syn_list.append(w_syn)
            self.w_bias_list.append(w_bias)

    def feed_forward(self, layer):
        for bias, weight in zip(self.w_bias_list, self.w_syn_list):
            layer = sigmoid(np.dot(weight, layer) + bias)
        return layer

    def trainner(self, input, size_input, output, epoc_max):
        
        l_list = []
        l_delta_list = []

        for j in range(epoc_max):
            iva = j % size_input
            ivb = iva + 1

            l0 = input[iva : ivb].T
            result = output[iva : ivb].T

            num_camadas = len(self.w_bias_list)

            layer = sigmoid(np.dot(self.w_syn_list[0], l0) + self.w_bias_list[0])
            l_list.append(layer)

            for indice in range(1, num_camadas):
                layer = sigmoid(np.dot(self.w_syn_list[indice], l_list[indice-1]) + self.w_bias_list[indice])
                l_list.append(layer)

            l_erro = result - l_list[-1]
            if (j % 10000) == 0:
                print('Error:' + str(np.mean(np.abs(l_erro))))

            #ultima camada
            l_delta = l_erro * sigmoid_derivative(l_list[-1])
            self.w_bias_list[-1] += l_delta
            l_erro = l_delta.T.dot(self.w_syn_list[-1])
            self.w_syn_list[-1] += l_list[-2].dot(l_delta.T).T

            #n camada
            for indice in range(2, num_camadas - 1):
                l_delta = l_erro.T * sigmoid_derivative(l_list[-indice])
                self.w_bias_list[-indice + 1] += l_delta
                l_erro = l_delta.T.dot( self.w_syn_list[-indice + 1] )
                self.w_syn_list[-indice +1] += l_list[-1].T.dot(l_delta)

            #primeira camada
            l_delta = l_erro.T * sigmoid_derivative(l_list[0])
            self.w_bias_list[0] += l_delta
            self.w_syn_list[0] += l0.dot(l_delta.T).T


