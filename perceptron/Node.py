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

from perceptron.Sinapse import Sinapse

class Node(object):
    '''
    Classe de Neoronio
    '''
    def __init__(self, id):
        #identificacao
        self.id = id

        #contem os pessos
        self.sinapses = []

        #sinapse de bias
        self.add_sinapse(Sinapse('bias', 1))

        #limiar
        self.threshold = 1

        self.taxa_aprendizado = 1

    def new_sinapse(self, id, entrada):
        '''
        Cria uma nova sinapse com id
        id: identificador da sinapse
        '''
        self.sinapses.append(Sinapse(id, entrada))

    def add_sinapse(self, sinapse):
        '''
        Adiciona uma sinapse criada ao neoronio
        sinapse: sinapse associada
        '''
        self.sinapses.append(sinapse)

    def get_sinapse(self, id):
        '''
        Retorna a sinapse pelo id
        id: id da sinapse ou None se ela nao existir
        '''
        for sinapse in self.sinapses:
            if sinapse.id == id:
                return sinapse

        return None

    def set_value_sinapse(self, id, valor):
        '''
        Ajusta uma nova entrada a sinapse
        id: id da sinapse a ter a entrada carregada
        valor: valor que sera caragado na sinapse nomeada peo id
        '''
        self.get_sinapse(id).entrada = valor

    def execute(self):
        '''
        Executa o Neoronio com os pesos das sinapses carregadas
        '''
        #fase 1 Ativacao
        y_in = 0
        for sinapse in self.sinapses:
            y_in += sinapse.ativacao()

        #fase 2 Propagacao
        if y_in > self.threshold:
            return 1
        elif y_in >= -self.threshold and y_in <= self.threshold:
            return 0
        else:
            return -1

    def treinar(self, resposta):
        '''
        Executa o treinamento
        do neoronio
        resposta: resposta deste treinamento expecifico
        '''
        result = self.execute()
        if result == resposta:
            return True
        else:
            for sinapse in self.sinapses:
                sinapse.ajusta_pesos(self.taxa_aprendizado, resposta)

            return False