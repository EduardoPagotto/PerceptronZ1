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

class Sinapse(object):
    '''
    Classe de sinapses controla os pesos e as entradas
    '''
    def __init__(self, id, entrada):
        #peso da sinapse
        self.peso = 0

        #valor de entrada
        self.entrada = entrada

        #identificador da sinapse
        self.id = id

    def ativacao(self):
        '''
        executa a funcao de ativacao da sinapse
        '''
        return self.entrada * self.peso

    def ajusta_pesos(self, taxa, resposta):
        '''
        equacao de ajustes de pesos
        taxa: taxa de aprendizagen
        resposta: resposta correta do treinamento
        '''
        self.peso += (taxa * resposta * self.entrada)