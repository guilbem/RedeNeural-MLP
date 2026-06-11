import numpy as np

# Calcula a função de ativação sigmoide binária
def sigmoide_binaria(x):

# Retorna valores no intervalo de 0 a 1
    return 1 / (1 + np.exp(-x))

# Calcula a derivada da sigmoide binária
# Utilizada durante o algoritmo Backpropagation
def derivada_sigmoide_binaria(x):

    # Calcula a saída da sigmoide
    s = sigmoide_binaria(x)

    # Aplica a fórmula da derivada
    return s * (1 - s)


# Calcula a função de ativação sigmoide bipolar
def sigmoide_bipolar(x):

# Retorna valores no intervalo de -1 a 1
    return (2 / (1 + np.exp(-x)) ) - 1


# Calcula a derivada da sigmoide bipolar
# Necessária para a atualização dos pesos
def derivada_sigmoide_bipolar(x):

    # Obtém a saída da sigmoide bipolar
    y = sigmoide_bipolar(x)

    # Aplica a fórmula da derivada
    return 0.5 * (1 + y) * (1 - y)


# Calcula a função tangente hiperbólica (tanh)
def tangente_hiperbolica(x):

    # Retorna valores no intervalo de -1 a 1
    return np.tanh(x)


# Calcula a derivada da tangente hiperbólica
# Utilizada durante a retropropagação do erro
def derivada_tangente_hiperbolica(x):

    # Retorna valores no intervalo de -1 a 1
    return 1 - np.tanh(x) ** 2