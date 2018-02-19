#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 20180219
Update on 20180219
@author: Eduardo Pagotto
'''

#pylint: disable=C0301
#pylint: disable=C0103
#pylint: disable=W0703
#pylint: disable=R0913

import numpy as np

from NeuralNet import NeuralNet
from ActivactionFunction import ActivationFunction, Sigmoid, Gaussian, Tanh

def teste_img():
    print('Teste IMG')

    #lista treinamento
    lista_v = np.array([ [0, 0, 1, 0,
                          0, 0, 1, 0,
                          0, 0, 1, 0,
                          0, 0, 1, 0],
                         [1, 1, 1, 1,
                          0, 1, 1, 0,
                          0, 1, 1, 0,
                          1, 1, 1, 1],
                         [1, 1, 1, 1,
                          1, 0, 0, 1,
                          1, 0, 0, 1,
                          1, 1, 1, 1],
                         [0, 0, 0, 0,
                          1, 1, 1, 1,
                          1, 0, 0, 1,
                          0, 0, 0, 0]])

    #lista resposta
    lista_r = np.array([ [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0] ])

    #16 entrada 4 saida composta de 3 hidden layer com 32, 24 e 8 neuronios
    grid=[16, 32, 24, 8, 4]
    neural = NeuralNet(grid, Sigmoid())
    neural.trainner(lista_v, 4, lista_r, 60000)
    neural.save('reconhe_padrao_img','padrao_img.json')

    neural2 = NeuralNet(None, Sigmoid())
    neural2.load('reconhe_padrao_img','padrao_img.json')

    for indice in range(4):
        li = lista_v[indice : indice + 1].T
        lo =  np.around(neural2.feed_forward(li), 1)
        print(lo.astype(int).T)

def teste_xor():

    print('Teste XOR')
    #lista treinamento
    lista_v = np.array([ [0, 0,], [0, 1], [1, 0 ], [1, 1 ]])

    #lista resposta
    lista_r = np.array([ [0, 0,], [1, 1], [1, 1 ], [0, 0 ]])

    #2 entrada 2 saida composta de 1 hidden layer com 3 neuronios
    grid=[2, 3, 2]
    neural = NeuralNet(grid, Sigmoid())
    neural.trainner(lista_v, 4, lista_r, 60000)
    neural.save('reconhe_padrao_xor','padrao_xor.json')

    neural2 = NeuralNet(None, Sigmoid())
    neural2.load('reconhe_padrao_xor','padrao_xor.json')

    for indice in range(4):
        li = lista_v[indice : indice + 1].T
        #print(neural2.feed_forward(li))
        lo =  np.around(neural2.feed_forward(li), 1)
        print(lo.astype(int).T)

if __name__ == '__main__':

    teste_xor()
    teste_img()