import numpy as np

import matplotlib.pyplot as plt

from src.funcao_ativacao import (
    sigmoide_binaria,
    sigmoide_bipolar,
    tangente_hiperbolica
)

# cria valores de x entre -10 e 10
# utilizados para desenhar as curvas
x = np.linspace(-10, 10, 1000)



# sigmoide binaria

# calcula os valores da função
y_sigmoide = sigmoide_binaria(x)

# cria uma nova figura
plt.figure()

# desenha a curva da função
plt.plot(x, y_sigmoide)

# título do gráfico
plt.title("Função Sigmoide Binária")

# nome eixo X
plt.xlabel("x")

# nome eixo Y
plt.ylabel("f(x)")

# adiciona grade ao gráfico
plt.grid()

# exibe gráfico
plt.show()



# sigmoide bipolar

# calcula os valores da função
y_bipolar = sigmoide_bipolar(x)

# cria nova figura
plt.figure()

# desenha curva
plt.plot(x, y_bipolar)

# título
plt.title("Função Sigmoide Bipolar")

# eixo X
plt.xlabel("x")

# eixo Y
plt.ylabel("f(x)")

# adiciona grade
plt.grid()

# exibe gráfico
plt.show()



# tangente hiperbolica

# calcula os valores da função
y_tanh = tangente_hiperbolica(x)

# cria nova figura
plt.figure()

# desenha curva
plt.plot(x, y_tanh)

# título
plt.title("Função Tangente Hiperbólica")

# eixo X
plt.xlabel("x")

# eixo Y
plt.ylabel("f(x)")

# adiciona grade
plt.grid()

# exibe gráfico
plt.show()