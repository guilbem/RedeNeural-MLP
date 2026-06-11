import numpy as np
import matplotlib.pyplot as plt

from src.mlp import mlp


# padrões XOR
X = np.array(

    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
)


# saídas desejadas
y = np.array(

    [
        [0],
        [1],
        [1],
        [0]
    ]
)


# cria rede
rede = mlp(

    entradas=2,

    ocultos=4,

    saidas=1,

    taxa=0.2
)


# erro tolerado
tolerancia = 0.001


# máximo de épocas
max_epocas = 100


# guarda erro por época
erros = []


print()
print("Iniciando treinamento...")
print()


for epoca in range(max_epocas):

    # FORWARD
    saida = rede.forward(X)

    # erro absoluto total
    erro_total = np.sum(np.abs(y - saida))

    # salva histórico
    erros.append(erro_total)

    # BACKWARD
    rede.backward(X, y)

    # mostra progresso
    if epoca % 100 == 0:

        print(f"Época {epoca} | Erro = {erro_total:.6f}")

    # verifica convergência
    if erro_total < tolerancia:

        print()
        print(f"Convergência atingida na época {epoca}")
        print()

        break


print("..............")
print("Treinamento Finalizado")


print()

print("Erro Final:", round(erro_total, 6))

print()

print("Épocas utilizadas:", epoca)

print()


# teste final
saida_final = rede.forward(X)

print("Resultados XOR")
print("..............")

for entrada, resultado in zip(X, saida_final):

    print(f"{entrada} -> {resultado[0]:.6f}")


# gráfico
plt.figure(figsize=(8,5))

plt.plot(erros)

plt.title("Erro x Época")

plt.xlabel("Épocas")

plt.ylabel("Erro Absoluto Total")

plt.grid(True)

plt.show()