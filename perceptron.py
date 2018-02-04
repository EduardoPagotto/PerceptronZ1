#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#pylint: disable=C0301
#pylint: disable=C0103
#pylint: disable=W0703

from perceptron.Node import Node
from perceptron.Perceptron import Perceptron

if __name__ == '__main__':

    nomes_sinapses = ['s0', 's1', 's2', 's3']
    respostas_corretas = [1, 1, 1, -1]
    entradas = [[-1, -1, 1, 1], [1, 1, 1, 1], [1, 1, -1, 1], [-1, -1, -1, 1]]
    #nomes_sinapses = ['s0','s1']
    #respostas_corretas = [1, -1, -1, 1]
    #entradas = [[-1, -1], [-1, 1], [1, -1], [ 1, 1]]

    node = Node('n0')

    for nome in nomes_sinapses:
        node.new_sinapse(nome, 0)

    perseptron = Perceptron()
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
