import numpy as np
class mlp:

    # construtor da rede
    def __init__(

        self,

        entradas=2,

        ocultos=4,

        saidas=1,

        taxa=0.2
    ):

        # salva taxa de aprendizagem
        self.taxa = taxa

        # cria pesos entre entrada e camada oculta
        self.pesos_entrada_oculta = np.random.uniform(-1, 1, (entradas, ocultos))

        # cria pesos entre camada oculta e saída
        self.pesos_oculta_saida = np.random.uniform(-1, 1, (ocultos, saidas))

        # cria bias da camada oculta
        self.bias_oculta = np.random.uniform(-1, 1, (1, ocultos))

        # cria bias da camada de saída
        self.bias_saida = np.random.uniform(-1, 1, (1, saidas))


    # função sigmóide
    def sigmoid(self, x):

        # aplica função sigmóide
        return 1 / (1 + np.exp(-x))

    # derivada da sigmóide
    def derivada_sigmoid(self, x):

        # calcula derivada
        return x * (1 - x)


    # etapa FORWARD
    def forward(self, X):

        # calcula entrada da camada oculta
        self.soma_oculta = np.dot(X, self.pesos_entrada_oculta) + self.bias_oculta

        # ativa camada oculta
        self.saida_oculta = self.sigmoid(self.soma_oculta)

        # calcula entrada da camada de saída
        self.soma_saida = np.dot(self.saida_oculta, self.pesos_oculta_saida) + self.bias_saida

        # ativa saída final
        self.saida_final = self.sigmoid(self.soma_saida)

        # retorna resultado
        return self.saida_final
    


    # etapa BACKWARD
    def backward(self, X, y):

        # calcula erro da saída
        erro_saida = y - self.saida_final

        # calcula delta da saída
        delta_saida = erro_saida * self.derivada_sigmoid(self.saida_final)

        # propaga erro para camada oculta
        erro_oculta = np.dot(delta_saida, self.pesos_oculta_saida.T)

        # calcula delta da camada oculta
        delta_oculta = erro_oculta * self.derivada_sigmoid(self.saida_oculta)

        # atualiza pesos oculta saída
        self.pesos_oculta_saida += self.taxa * np.dot(self.saida_oculta.T, delta_saida)

        # atualiza bias saída
        self.bias_saida += self.taxa * np.sum(delta_saida, axis=0, keepdims=True)

        # atualiza pesos entrada oculta
        self.pesos_entrada_oculta += self.taxa * np.dot(X.T, delta_oculta)

        # atualiza bias oculta
        self.bias_oculta += self.taxa * np.sum(delta_oculta, axis=0, keepdims=True)


          # calcula erro absoluto total
    def calcular_erro(self, y):

        return np.sum(np.abs(y - self.saida_final))