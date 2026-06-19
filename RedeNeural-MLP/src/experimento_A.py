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
print("...........")
print("Experimento A")
print(".........")
print()


# quantidades de neurônios ocultos
quantidades = [

    2,
    3,
    4,
    5
]


# percorre cada configuração
for ocultos in quantidades:

    print()

    print(f"Testando {ocultos} neurônios ocultos")

    print("--------------------------")


    # cria rede
    rede = mlp(

        entradas=2,

        ocultos=ocultos,

        saidas=1,

        taxa=0.2
    )


    # parâmetros
    tolerancia = 0.001
    max_epocas = 1000


    # treinamento
    for epoca in range(max_epocas):

        # forward
        saida = rede.forward(X)

        # erro total
        erro_total = np.sum(np.abs(y - saida))

        # backward
        rede.backward(X, y)

        # verifica convergência
        if erro_total < tolerancia:

            break


    # resultados
    print(f"Épocas: {epoca}")

    print(f"Erro Final: {erro_total:.6f}")

    print()