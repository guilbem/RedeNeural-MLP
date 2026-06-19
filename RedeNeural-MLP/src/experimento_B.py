import numpy as np

from mlp import mlp

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


print()
print("-------------------")
print("Experimento B")

print()


# taxas a testar
taxas = [

    0.1,
    0.2,
    0.3,
    0.4,
    0.5
]


# percorre taxas
for taxa in taxas:

    print()

    print(f"Taxa de aprendizagem = {taxa}")

    print("--------------------------")


    # cria rede
    rede = mlp(

        entradas=2,

        ocultos=4,

        saidas=1,

        taxa=taxa
    )


    tolerancia = 0.001
    max_epocas = 1000


    # treinamento
    for epoca in range(max_epocas):

        saida = rede.forward(X)

        erro_total = np.sum(np.abs(y - saida))

        rede.backward(X, y)

        if erro_total < tolerancia:

            break


    print(f"Épocas: {epoca}")

    print(f"Erro Final: {erro_total:.6f}")

    print()