#!/usr/bin/env python
# -*- coding: utf-8 -*-

#pylint: disable=C0301
#pylint: disable=C0103
#pylint: disable=W0703

class Sinapse(object):
    '''
    Classe de sinapses controla os pesos e as entradas
    '''
    def __init__(self, id):
        #peso da sinapse
        self.peso = 0

        #valor de entrada
        self.entrada = 0

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

class Node(object):
    '''
    Classe de Neoronio
    '''
    def __init__(self, id):
        #identificacao
        self.id = id

        #contem os pesoa
        self.sinapses = []

        #ajuste fino
        self.bias = 0

        #limiar
        self.threshold = 1

        self.taxa_aprendizado = 1

    def new_sinapse(self, id):
        '''
        Cria uma nova sinapse com id
        id: identificador da sinapse
        '''
        self.sinapses.append(Sinapse(id))

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
        soma = 0
        for sinapse in self.sinapses:
            soma += sinapse.ativacao()

        #fase 2 Propagacao
        y_in = self.bias + soma
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

            self.bias += self.taxa_aprendizado * resposta

            return False


class Perseptron(object):
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
        Executa o treinamento dos pesos, e bias
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
                    node.set_value_sinapse(id_sinapses[j], dados[j])

                if node.treinar(respostas[indice]) is False:
                    print('Errou em {0}'.format(indice))
                else:
                    print('Acertou em {0}'.format(indice))
                    total_ok += 1

            if total_ok == 4:
                print('Rede Treinada')
                retorno = True
                break

        return retorno

if __name__ == '__main__':

    nomes_sinapses = ['s0', 's1', 's2', 's3']
    respostas_corretas = [1, 1, 1, -1]
    entradas = [[-1, -1, 1, 1], [1, 1, 1, 1], [1, 1, -1, 1], [-1, -1, -1, 1]]
    #nomes_sinapses = ['s0','s1']
    #respostas_corretas = [1, -1, -1, 1]
    #entradas = [[-1, -1], [-1, 1], [1, -1], [ 1, 1]]

    node = Node('n0')

    for nome in nomes_sinapses:
        node.new_sinapse(nome)

    perseptron = Perseptron()
    perseptron.add_node(node)

    if perseptron.treinamento(nomes_sinapses, entradas, respostas_corretas) is True:
        print('OK')

        node.set_value_sinapse('s0', 1)
        node.set_value_sinapse('s1', 1)
        val = node.execute()
        print('saida:{0}'.format(val))

        node.set_value_sinapse('s0', -1)
        node.set_value_sinapse('s1', 1)
        val = node.execute()
        print('saida:{0}'.format(val))

        node.set_value_sinapse('s0', 1)
        node.set_value_sinapse('s1', -1)
        val = node.execute()
        print('saida:{0}'.format(val))

        node.set_value_sinapse('s0', -1)
        node.set_value_sinapse('s1', -1)
        val = node.execute()
        print('saida:{0}'.format(val))

    else:
        print('ERRO')

    print(perseptron)
