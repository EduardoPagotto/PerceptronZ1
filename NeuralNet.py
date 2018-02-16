#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 20180215
Update on 20180216
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
    '''
    Classe de Neuronios padrao camadas N0:N1+1:..:NN+1:NF
    '''
    def __init__(self, in_size, out_size, hidden_layers, hidden_size):
        '''
        Inicializa listas de pesos sinapticos e lista de bias com valores aleatorios
        in_size:numero de neuronios de entrada
        out_size:numero de neuronios de saida
        hidden_layers: numero de camadas
        hidden_size numero de neuronios na camada hidden_layers
        '''
        self.w_syn_list = []
        self.w_bias_list = []

        for index in range(hidden_layers + 1):
            
            #TODO optimizar no append direto
            if index == 0:
                w_syn = 2 * np.random.random((hidden_size, in_size)) - 1
                w_bias = 2 * np.random.random((hidden_size,1)) - 1

            elif index == hidden_layers:
                w_syn = 2 * np.random.random((out_size, hidden_size)) - 1
                w_bias = 2 * np.random.random((out_size,1)) - 1
            else:
                w_syn = 2 * np.random.random((hidden_size, hidden_size)) - 1
                w_bias = 2 * np.random.random((hidden_size,1)) - 1
        
            self.w_syn_list.append(w_syn)
            self.w_bias_list.append(w_bias)

    def save(self):
        '''
        Salva no diretorio corrente os pesos sinapticos e bias
        '''
        for indice in range(self.w_syn_list):
            np.savetxt('syn{0}.txt'.format(indice), self.w_syn_list[indice], fmt='%f')

        for indice in range(self.w_bias_list):
            np.savetxt('bias{0}.txt'.format(indice), self.w_bias_list[indice], fmt='%f')

    def feed_forward(self, layer):
        '''
        Percursor do grafo(neuronios)
        layer: matriz de dados de entrada
        return: matriz de dados de saida
        '''
        for bias, weight in zip(self.w_bias_list, self.w_syn_list):
            layer = sigmoid(np.dot(weight, layer) + bias)
        return layer

    def dep_layer_trainer(self, index, epoc, local_layer, limite, result):
        '''
        Rotina recursiva de treinamento back-forward
        index: indice da camada a processar
        epoc: contador de treinamento
        local_layer: neuronio pai
        limite: indice maximo
        result: resultado esperado
        return: erro da camada anterior ou None para primeira camada 
        '''
        erro = None
        layer = sigmoid(np.dot(self.w_syn_list[index], local_layer) + self.w_bias_list[index])

        if index < limite - 1:
            erro = self.dep_layer_trainer(index + 1, epoc, layer, limite, result)

            delta = erro.T * sigmoid_derivative(layer)
            self.w_bias_list[index] += delta

            if index != 0:
                erro = delta.T.dot(self.w_syn_list[index])
                self.w_syn_list[index] += local_layer.T.dot(delta)
            else:
                #primeira camada nao tem ocrrecao de erro para a proxima               
                self.w_syn_list[index] += local_layer.dot(delta.T).T
        else:
            
            erro = result - layer
            if (epoc % 10000) == 0:
                print('Error:' + str(np.mean(np.abs(erro))))

            delta = erro * sigmoid_derivative(layer)
            self.w_bias_list[index] += delta

            erro = delta.T.dot(self.w_syn_list[index])
            self.w_syn_list[index] += local_layer.dot(delta.T).T

        return erro

    def trainner(self, input, size_input, output, epoc_max):
        '''
        Executa um treinamento na rede
        input: matriz de dados entrada
        size_input: total de items por amostrage
        output: matriz de dados de resposta esperada
        epoc_max: numero maximo de treinamento
        '''
        for j in range(epoc_max):
            iva = j % size_input
            ivb = iva + 1

            l0 = input[iva : ivb].T
            result = output[iva : ivb].T

            num_camadas = len(self.w_bias_list)
            self.dep_layer_trainer(0, j, l0, num_camadas, result)

if __name__ == '__main__':

    #lista treinamento
    lista_v = np.array([ [0, 0], [0, 1],[1, 0], [1, 1] ])
    
    #lista resposta
    lista_r = np.array([ [1, 1], [0, 0],[0, 0], [1, 1] ])

    neural = NeuralNet(2,2,2,3)
    neural.trainner(lista_v, 4, lista_r, 60000)

    for indice in range(4):
        li = lista_v[indice : indice + 1].T        
        lo = neural.feed_forward(li)
        print(lo)

