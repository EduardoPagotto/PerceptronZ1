#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 20180204
Update on 20180215
@author: Eduardo Pagotto
'''

#pylint: disable=C0301
#pylint: disable=C0103
#pylint: disable=W0703
#pylint: disable=R0913

import numpy as np

def sigmoid(x):
    '''
    função sigmoid
    '''
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    '''
    função sigmoid derivada
    '''  
    return x*(1-x)

def tanh(x, derivative=False):
    if (derivative == True):
        return (1 - (x ** 2))
    return np.tanh(x)

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

def gaussian(x, derivative=False):
    if (derivative == True):
        for i in range(0, len(x)):
            for k in range(0, len(x[i])):
                x[i][k] = -2* x[i][k] * np.exp(-x[i][k] ** 2)
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            x[i][k] = np.exp(-x[i][k] ** 2)
    return x

def uma_camada():
    
    # input dataset
    X = np.array([  [0,0,1],
                    [0,1,1],
                    [1,0,1],
                    [1,1,1] ])
        
    # output dataset            
    y = np.array([[0,0,1,1]]).T

    # seed random numbers to make calculation
    # deterministic (just a good practice)
    #np.random.seed(1)

    # initialize weights randomly with mean 0
    syn0 = 2*np.random.random((3,1)) - 1

    for iter in range(10000):

        # forward propagation
        l0 = X
        l1 = sigmoid(np.dot(l0,syn0))

        # how much did we miss?
        l1_error = y - l1

        # multiply how much we missed by the 
        # slope of the sigmoid at the values in l1
        l1_delta = l1_error * sigmoid_derivative(l1)

        # update weights
        syn0 += np.dot(l0.T,l1_delta)

    print("Output After Training:")
    print(l1)

def duas_camadas():
    #input
    x = np.array([ [0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1] ])


    #output
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])


    #np.random.seed(1)
    #np.random.seed(1)

    #synapses
    syn0 = 2 * np.random.random((3,4)) - 1
    syn1 = 2 * np.random.random((4,1)) - 1
    
    #treinamento
    for j in range(60000):

        # Feed forward through layers 0, 1, and 2
        l0 = x
        l1 = sigmoid(np.dot(l0, syn0))
        l2 = sigmoid(np.dot(l1, syn1))

        # how much did we miss the target value?
        l2_error = y - l2

        if (j % 10000) == 0:
            print('Error:' + str(np.mean(np.abs(l2_error))))

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error * sigmoid_derivative(l2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.     
        l1_delta = l1_error * sigmoid_derivative(l1)

        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)

    print('Saida apos o treino')
    print(l2)

def multi_laye_xor():
    #synapses
    syn0 = 2 * np.random.random((3,2)) - 1
    syn1 = 2 * np.random.random((3,3)) - 1
    syn2 = 2 * np.random.random((2,3)) - 1

    #bias
    w_bias0 = 2 * np.random.random((3,1)) - 1
    w_bias1 = 2 * np.random.random((3,1)) - 1
    w_bias2 = 2 * np.random.random((2,1)) - 1

    lista_v = np.array([ [0, 0], [0, 1],[1, 0], [1, 1] ])
    lista_r = np.array([ [1, 1], [0, 0],[0, 0], [1, 1] ])

    for j in range(60000):

        iva = j % 4
        ivb = iva + 1

        l0 = lista_v[iva : ivb].T
        result = lista_r[iva : ivb].T

        l1 = sigmoid(np.dot(syn0, l0) + w_bias0)
        l2 = sigmoid(np.dot(syn1, l1) + w_bias1)
        l3 = sigmoid(np.dot(syn2, l2) + w_bias2)

        l3_erro = result - l3

        if (j % 10000) == 0:
            print('Error:' + str(np.mean(np.abs(l3_erro))))

        l3_delta = l3_erro * sigmoid_derivative(l3)

        l2_error = l3_delta.T.dot(syn2)
        l2_delta = l2_error.T * sigmoid_derivative(l2)

        l1_error = l2_delta.T.dot(syn1)
        l1_delta = l1_error.T * sigmoid_derivative(l1)

        syn2 += l2.dot(l3_delta.T).T
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.dot(l1_delta.T).T

        w_bias2 += l3_delta
        w_bias1 += l2_delta
        w_bias0 += l1_delta


    print('Teste Final.....')

    np.savetxt('syn0.txt', syn0, fmt='%f')
    np.savetxt('syn1.txt', syn1, fmt='%f')
    np.savetxt('syn2.txt', syn2, fmt='%f')

    np.savetxt('bias0.txt', w_bias0, fmt='%f')
    np.savetxt('bias1.txt', w_bias1, fmt='%f')
    np.savetxt('bias2.txt', w_bias2, fmt='%f')

    #teste = np.loadtxt('syn0.txt', dtype=float)

    for indice in range(4):
        l0 = lista_v[indice : indice + 1].T
        l1 = sigmoid(np.dot(syn0, l0) + w_bias0)
        l2 = sigmoid(np.dot(syn1, l1) + w_bias1)
        l3 = sigmoid(np.dot(syn2, l2) + w_bias2)

        print(l3)
    
    print('FIM..') 

if __name__ == '__main__':

    multi_laye_xor()