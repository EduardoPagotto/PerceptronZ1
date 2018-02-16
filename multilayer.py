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
from ActivactionFunction import sigmoid
from ActivactionFunction import sigmoid_derivative

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

def feed_forward(layer, lista_bias, lista_syn):
    for bias, weight in zip(lista_bias, lista_syn):
        layer = sigmoid(np.dot(weight, layer) + bias)
    return layer

def multi_laye_xor():
    #pesos sinapticos
    syn = []
    syn.append(2 * np.random.random((3,2)) - 1)
    syn.append(2 * np.random.random((3,3)) - 1)
    syn.append(2 * np.random.random((2,3)) - 1)

    #pesos bias
    w_bias = []
    w_bias.append(2 * np.random.random((3,1)) - 1)
    w_bias.append(2 * np.random.random((3,1)) - 1)
    w_bias.append(2 * np.random.random((2,1)) - 1)

    #lista treinamento
    lista_v = np.array([ [0, 0], [0, 1],[1, 0], [1, 1] ])
    
    #lista resposta
    lista_r = np.array([ [1, 1], [0, 0],[0, 0], [1, 1] ])

    for j in range(60000):

        iva = j % 4
        ivb = iva + 1

        input = lista_v[iva : ivb].T
        result = lista_r[iva : ivb].T

        l = []
        l.append(sigmoid(np.dot(syn[0], input) + w_bias[0]))
        l.append(sigmoid(np.dot(syn[1], l[0]) + w_bias[1]))
        l.append(sigmoid(np.dot(syn[2], l[1]) + w_bias[2]))

        erro = result - l[2]

        if (j % 10000) == 0:
            print('Error:' + str(np.mean(np.abs(erro))))

        delta = erro * sigmoid_derivative(l[2])
        w_bias[2] += delta
        
        error = delta.T.dot(syn[2])
        syn[2] += l[1].dot(delta.T).T

        delta = error.T * sigmoid_derivative(l[1])
        w_bias[1] += delta
        error = delta.T.dot(syn[1])
        syn[1] += l[0].T.dot(delta)

        delta = error.T * sigmoid_derivative(l[0])
        w_bias[0] += delta

        syn[0] += input.dot(delta.T).T

    print('Teste Final.....')

    np.savetxt('syn0.txt', syn[0], fmt='%f')
    np.savetxt('syn1.txt', syn[1], fmt='%f')
    np.savetxt('syn2.txt', syn[2], fmt='%f')

    np.savetxt('bias0.txt', w_bias[0], fmt='%f')
    np.savetxt('bias1.txt', w_bias[1], fmt='%f')
    np.savetxt('bias2.txt', w_bias[2], fmt='%f')
    #teste = np.loadtxt('syn0.txt', dtype=float)

    for indice in range(4):
        input = lista_v[indice : indice + 1].T        
        l3 = feed_forward(input, w_bias, syn)
        print(l3)
    
    print('FIM..') 




if __name__ == '__main__':

    multi_laye_xor()