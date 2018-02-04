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

from perceptron.Node import Node

class Perceptron(object):
    '''
    Classe acesso aos nodes
    '''
    def __init__(self):
        self.lista_Node = []

    def add_node(self, node):
        '''
        Adiciona um node ao perceptron
        node: node associado
        '''
        self.lista_Node.append(node)

    def treinamento(self, id_sinapses, entrada_dados, respostas):
        '''
        Executa o treinamento dos pesos incluindo uma sinapse de bias
        id_sinapses: lista com as identificacoes das sinapses a serem usadas
        entrada_dados: array de array de dados de entrata do trinamento
        respostas: array com a lista de respostas corretas a cada interacao
        '''
        retorno = False
        for interacao in range(0, 239):
            total_ok = 0
            print('ciclo----------- {0}'.format(interacao))
            for indice in range(0, len(entrada_dados)):
                dados = entrada_dados[indice]
                for j in range(0, len(dados)):
                    self.lista_Node[0].set_value_sinapse(id_sinapses[j], dados[j])

                if self.lista_Node[0].treinar(respostas[indice]) is False:
                    print('Errou em {0}'.format(indice))
                else:
                    print('Acertou em {0}'.format(indice))
                    total_ok += 1

            if total_ok == 4:
                print('Rede Treinada')
                retorno = True
                break

        return retorno