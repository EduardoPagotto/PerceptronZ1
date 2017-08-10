#!/usr/bin/env python
# -*- coding: utf-8 -*-

#pylint: disable=C0301
#pylint: disable=C0103
#pylint: disable=W0703

# aplicativo para verificar se o ser vivo eh quadrupede ou bipede
# quadrupede = 1, bipede = -1
# cao = [-1,-1,1,1] | resposta = 1
# gato = [1,1,1,1] | resposta = 1
# cavalo = [1,1,-1,1] | resposta = 1
# homem = [-1,-1,-1,1] | resposta = -1

# pesos (sinapses)
sinapses = [0, 0, 0, 0]

# entradas
entradas = [[-1, -1, 1, 1], [1, 1, 1, 1], [1, 1, -1, 1], [-1, -1, -1, 1]]

# respostas esperadas
respostas = [1, 1, 1, -1]

# bias (ajuste fino)
bias = 0

#saida
saida = 0

# numero maximo de interacoes
max_int = 10

# taxa de aprendizado
taxa_aprendizado = 1

#soma
soma = 0

#theshold
threshold = 1

# nome do animal
animal = ""

# resposta = acerto ou falha
resposta = ""

# dicionario de dados
d = {'-1,-1,1,1' : 'cao',
     '1,1,1,1' : 'gato',
     '1,1,-1,1' : 'cavalo',
     '-1,-1,-1,1' : 'homem'}

print("Treinando")

# funcao para converter listas em strings
def listToString(lista):
    '''pega dados apenas'''
    s = str(lista).strip('[]')
    s = s.replace(' ', '')
    return s

# inicio do algoritmo
for k in range(1, max_int):
    acertos = 0
    print("INTERACAO "+str(k)+"-------------------------")
    for i in range(0, len(entradas)):
        soma = 0

        # pega o nome do animal no dicionário
        v1 = listToString(entradas[i])
        #if d.has_key(v1):
        if v1 in d:
            animal = d[listToString(entradas[i])]
        else:
            animal = ""

        # para calcular a saida do perceptron, cada entrada de x eh multiplicada
        # pelo seu peso w correspondente
        for j in range(0, len(entradas[i])):
            soma += entradas[i][j] * sinapses[j]

        # a saida eh igual a adicao do bias com a soma anterior
        y_in = bias + soma
        #print("y_in = ",str(y_in))

        # funcao de saida eh determinada pelo threshold
        if y_in > threshold:
            saida = 1
        elif y_in >= -threshold and y_in <= threshold:
            saida = 0
        else:
            saida = -1

        # atualiza os pesos caso a saida nao corresponda ao valor esperado
        if saida == respostas[i]:
            acertos += 1
            resposta = "acerto"
        else:
            for j in range(0, len(sinapses)):
                sinapses[j] = sinapses[j] + (taxa_aprendizado * respostas[i] * entradas[i][j])
            bias = bias + taxa_aprendizado * respostas[i]
            resposta = "Falha - Peso atualizado"

        #imprime a resposta
        if saida == 1:
            print(animal+" = quadrupede = "+resposta)
        elif saida == 0:
            print(animal+" = padrao nao identificado = "+resposta)
        elif saida == -1:
            print(animal+" = bipede = "+resposta)

    if acertos == len(entradas):
        print("Funcionalidade aprendida com "+str(k)+" interacoes")
        break
    print("")
print("Finalizado")

