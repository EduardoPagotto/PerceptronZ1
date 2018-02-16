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
    Classe de Neuronios padrao camadas definidas por grid
    '''
    def __init__(self, grid):
        '''
        Inicializa listas de pesos sinapticos e lista de bias com valores aleatorios
        grid: array com geometrida das camadas de entrada, hidden's e saida
        [entrada, h1, h2, hn , saida]
        '''
        self.w_syn_list = []
        self.w_bias_list = []

        for index in range(1, len(grid)):
            
            atual = grid[index]
            anterior = grid[index-1]

            w_syn = 2 * np.random.random((atual, anterior)) - 1
            w_bias = 2 * np.random.random((atual, 1)) - 1           
        
            self.w_syn_list.append(w_syn)
            self.w_bias_list.append(w_bias)

    def save(self):
        '''
        Salva no diretorio corrente os pesos sinapticos e bias
        '''
        for indice in range(len(self.w_syn_list)):
            np.savetxt('syn{0}.txt'.format(indice), self.w_syn_list[indice], fmt='%f')

        for indice in range(len(self.w_bias_list)):
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
        #variavel de retorno de erro para acamada inferior
        erro = None
        layer = sigmoid(np.dot(self.w_syn_list[index], local_layer) + self.w_bias_list[index])

        if index < limite - 1:
            #camadas 0 ate penultima
            erro = self.dep_layer_trainer(index + 1, epoc, layer, limite, result)

            #calcula o delta atual pelo erro da camada anterior e acumula no bias
            delta = erro.T * sigmoid_derivative(layer)
            self.w_bias_list[index] += delta

            if index != 0:
                #demais camadas calcula o erro local pelo delta da camada 
                erro = delta.T.dot(self.w_syn_list[index])

                #calcula baseado no shapes passados
                if self.w_bias_list[index].shape[0] != self.w_bias_list[index -1].shape[0]:
                    self.w_syn_list[index] += local_layer.dot(delta.T).T
                else:
                    self.w_syn_list[index] += local_layer.T.dot(delta)
            else:
                #primeira camada nao tem correção de erro para a proxima               
                self.w_syn_list[index] += local_layer.dot(delta.T).T
        else:
            #Ultima camada calcula o delta e o erro para a camada anterior
            erro = result - layer
            if (epoc % 10000) == 0:
                print('Error:' + str(np.mean(np.abs(erro))))
            
            #multiplica o erro pela derivada da camada atual
            delta = erro * sigmoid_derivative(layer)

            #acumula o delta no Bias
            self.w_bias_list[index] += delta

            #erro e o produto do delta com os pesos sinapticos atual
            #calcula baseado no shapes passados
            if delta.shape[1] != self.w_syn_list[index].shape[0]:
                erro = delta.T.dot(self.w_syn_list[index])
            else:
                erro = delta.dot(self.w_syn_list[index])

            #acumula pelos na camada anterior
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

    #16 entrada 4 saida 1 hidden layer com 3 camadas 
    grid=[16, 32, 24, 8, 4]
    neural = NeuralNet(grid)
    neural.trainner(lista_v, 4, lista_r, 60000)

    neural.save()

    for indice in range(4):
        li = lista_v[indice : indice + 1].T        
        lo = (10 * neural.feed_forward(li)) / 9
        print(lo.astype(int).T)



