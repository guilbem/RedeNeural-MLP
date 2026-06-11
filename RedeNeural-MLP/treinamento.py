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
max_epocas = 10000


# guarda erro por época
erros = []


# controla convergência
convergiu = False


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

        print( f"Época {epoca} | "f"Erro = {erro_total:.6f}")

    # verifica convergência
    if erro_total < tolerancia:

        convergiu = True

        print()

        print(f"Convergência atingida na época {epoca}")

        print()

        break


print("..............")
print("Treinamento Finalizado")

print()


# verifica resultado final
if convergiu:

    print("Status: Rede convergiu.")

else:

    print("Status: Rede NÃO convergiu.")


print()

print("Erro Final:", round(erro_total, 6)
)

print()

print("Menor erro atingido:", round(min(erros), 6))

print()

print("Épocas utilizadas:",epoca) 

print()


# teste final
saida_final = rede.forward(X)

print("Resultados XOR")
print("..............")

for entrada, esperado, resultado in zip(X, y,saida_final):

    print(

        f"Entrada: {entrada} "
        f"| Esperado: {esperado[0]} "
        f"| Obtido: {resultado[0]:.6f}"
    )


print()


# gráfico erro x época
plt.figure(figsize=(8, 5))

plt.plot(erros, label="Erro Total")

plt.title("Erro x Época")

plt.xlabel("Épocas")

plt.ylabel("Erro Absoluto Total")

plt.legend()

plt.grid(True)

plt.show()