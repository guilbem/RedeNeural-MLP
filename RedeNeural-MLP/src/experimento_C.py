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
print(".................")
print("Experimento C")
print("Influência dos Pesos Iniciais")
print(".................")
print()


# sementes
sementes = [

    0,
    1,
    2,
    3,
    4
]


for seed in sementes:

    print()

    print(f"Semente = {seed}")

    print("--------------------")


    # fixa aleatoriedade
    np.random.seed(seed)


    # cria rede
    rede = mlp(

        entradas=2,

        ocultos=4,

        saidas=1,

        taxa=0.2
    )


    tolerancia = 0.001
    max_epocas = 10000


    for epoca in range(max_epocas):

        saida = rede.forward(X)

        erro_total = np.sum(np.abs(y - saida))

        rede.backward(X,y)

        if erro_total < tolerancia:

            break


    print(f"Épocas: {epoca}")

    print(f"Erro Final: {erro_total:.6f}")

    print()