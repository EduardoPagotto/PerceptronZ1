#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 20180215
Update on 20180219
@author: Eduardo Pagotto
'''

#pylint: disable=C0301
#pylint: disable=C0103
#pylint: disable=W0703
#pylint: disable=R0913

import numpy as np

class ActivationFunction:
    def forward(self, x):
        pass
    def backward(self, x):
        pass

class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1.0/(1.0+np.exp(-x))

    def backward(self, x):
        return x*(1.0 - x)

class Gaussian(ActivationFunction):
    def forward(self, x):
        for i in range(0, len(x)):
            for k in range(0, len(x[i])):
                x[i][k] = np.exp(-x[i][k] ** 2)
        return x

    def backward(self, x):
        for i in range(0, len(x)):
            for k in range(0, len(x[i])):
                x[i][k] = -2* x[i][k] * np.exp(-x[i][k] ** 2)
        return x

class Tanh(ActivationFunction):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return (1 - (x ** 2))

def relu(x, derivative=False):
    if (derivative == True):
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 1
                else:
                    x[i][k] = 0
        return x
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                pass  # do nothing since it would be effectively replacing x with x
            else:
                x[i][k] = 0
    return x

def arctan(x, derivative=False):
    if (derivative == True):
        return (np.cos(x) ** 2)
    return np.arctan(x)

def step(x, derivative=False):
    if (derivative == True):
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 0
        return x
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                x[i][k] = 1
            else:
                x[i][k] = 0
    return x

def squash(x, derivative=False):
    if (derivative == True):
        for i in range(0, len(x)):
            for k in range(0, len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = (x[i][k]) / (1 + x[i][k])
                else:
                    x[i][k] = (x[i][k]) / (1 - x[i][k])
        return x
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            x[i][k] = (x[i][k]) / (1 + abs(x[i][k]))
    return x

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
