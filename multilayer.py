#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 20180204
Update on 20180204
@author: Eduardo Pagotto
'''

#pylint: disable=C0301
#pylint: disable=C0103
#pylint: disable=W0703
#pylint: disable=R0913

import numpy as np
#import math

def nonlin(x, deriv=False):
    '''
    função sigmoid
    '''
    if deriv is True:
        return x*(1-x)

    return 1/(1+np.exp(-x))

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
        l1 = nonlin(np.dot(l0,syn0))

        # how much did we miss?
        l1_error = y - l1

        # multiply how much we missed by the 
        # slope of the sigmoid at the values in l1
        l1_delta = l1_error * nonlin(l1,True)

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
        l1 = nonlin(np.dot(l0, syn0))
        l2 = nonlin(np.dot(l1, syn1))

        # how much did we miss the target value?
        l2_error = y - l2

        if (j % 10000) == 0:
            print('Error:' + str(np.mean(np.abs(l2_error))))

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error * nonlin(l2, deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.     
        l1_delta = l1_error * nonlin(l1, deriv=True)

        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)

    print('Saida apos o treino')
    print(l2)


if __name__ == '__main__':

    #synapses
    syn0 = 2 * np.random.random((3,2)) - 1
    syn1 = 2 * np.random.random((3,3)) - 1
    syn2 = 2 * np.random.random((2,3)) - 1

    #l0 = np.array([ [0],[0] ])

    lista_v = [
        np.array([ [0],[0] ]),
        np.array([ [0],[1] ]),
        np.array([ [1],[0] ]),
        np.array([ [1],[1] ]),
    ]

    lista_r = [
        np.array([ [1],[1] ]),
        np.array([ [1],[0] ]),
        np.array([ [0],[1] ]),
        np.array([ [0],[0] ]),
    ]

    #result = np.array([ [0], [0] ])
    for j in range(500000):

        iv = j % 4
        ir = j % 4

        l0 = lista_v[iv]
        result = lista_r[ir]

        l1 = nonlin(np.dot(syn0, l0))
        l2 = nonlin(np.dot(syn1, l1))
        l3 = nonlin(np.dot(syn2, l2))

        l3_erro = result - l3

        if (j % 10000) == 0:
            print('Error:' + str(np.mean(np.abs(l3_erro))))

        l3_delta = l3_erro * nonlin(l3, deriv=True)

        l2_error = l3_delta.T.dot(syn2)
        l2_delta = l2_error.T * nonlin(l2, deriv=True)

        l1_error = l2_delta.T.dot(syn1)
        l1_delta = l1_error.T * nonlin(l1, deriv=True)

        syn2 += l2.dot(l3_delta.T).T
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.dot(l1_delta.T).T

    print('Teste Final.....')

    for val in lista_v:
        #lt = np.array([ [1],[0] ])
        l1 = nonlin(np.dot(syn0, val))
        l2 = nonlin(np.dot(syn1, l1))
        l3 = nonlin(np.dot(syn2, l2))

        print(l3)