import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# fixa semente aleatória
np.random.seed(42)

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

    taxa=0.5
)

# erro tolerado
tolerancia = 0.1

# máximo de épocas
max_epocas = 618

# guarda erro por época
erros = []

# controla convergência
convergiu = False

print()
print("Iniciando treinamento...")
print()

# laço principal
for epoca in range(max_epocas):

    # zera erro da época
    erro_total = 0

    # percorre cada padrão XOR
    for entrada, esperado in zip(X, y):

        # transforma em matriz 1x2
        entrada = entrada.reshape(1, -1)

        # transforma em matriz 1x1
        esperado = esperado.reshape(1, -1)

        # FORWARD
        saida = rede.forward(entrada)

        # calcula erro quadrático médio
        erro = np.mean((esperado - saida) ** 2)

        # acumula erro
        erro_total += erro

        # BACKWARD
        rede.backward(entrada,esperado)


    # salva histórico
    erros.append(erro_total)


    # mostra progresso
    if epoca % 100 == 0:

        print(f"Época {epoca} | "f"Erro = {erro_total:.8f}")


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

print("Erro Final:", round(erro_total, 8))

print()

print("Menor erro atingido:", round(min(erros), 8))

print()

print("Épocas utilizadas:", epoca)


print()


# teste final
saida_final = rede.forward(X)

print("Resultados XOR")
print("..............")


for entrada, esperado, resultado in zip(X, y, saida_final):

    print(f"Entrada: {entrada} "f"| Esperado: {esperado[0]} "f"| Obtido: {resultado[0]:.6f}")


print()

# gráfico erro x época
plt.figure(figsize=(8, 5))

plt.plot(erros, label="Erro Total")

plt.title("Erro x Época")

plt.xlabel("Épocas")

plt.ylabel("Erro Quadrático Total")

plt.legend()

plt.grid(True)

plt.show()


# cria pasta resultados
os.makedirs("resultados", exist_ok=True)


# cria tabela de erros
df = pd.DataFrame({"Epoca": range(len(erros)),"Erro": erros})


# salva csv
df.to_csv("resultados/erro_por_epoca.csv", index=False)


print()

print("Arquivo salvo em:")

print("resultados/erro_por_epoca.csv")

print()