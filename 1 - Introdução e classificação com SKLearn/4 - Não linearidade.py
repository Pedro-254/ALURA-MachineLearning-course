# from sklearn.svm import SVC
# import numpy as np
# from sklearn.model_selection import train_test_split

# SEED = 5
# np.random.seed(SEED)
# treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,
#                                                          stratify = y)
# print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

# modelo = SVC(gamma='auto')

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# Importação dos dados CSV
uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)

print(dados.head())

renomeacao = {
    "unfinished" : "nao_finalizado",
    "expected_hours" : "horas_esperadas",
    "price" : "preco",
}

dados = dados.rename(columns = renomeacao)
print(dados.head())

troca = {
    0 : 1,
    1 : 0
}

dados['finalizado'] = dados.nao_finalizado.map(troca)
print(dados.tail())
    
x = dados[["horas_esperadas","preco"]]
y = dados["finalizado"]

print("Quantidade de dados: Linhas: %d Colunas: %d "% (dados.shape[0], dados.shape[1]))


SEED = 5
# Setando a seed do np para ser a SEED tornando o modelo (que usa essa seed) fixo!
np.random.seed(SEED)
modelo = SVC(gamma='auto')

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y,random_state = SEED, test_size = 0.25,stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(raw_treino_x), len(raw_teste_x)))

# Baseado no treino_x criase uma escala nova e atribui essa escala para o treino_x e teste_x
scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)


previsoes_de_base = np.ones(540)
acuracia = accuracy_score(teste_y, previsoes_de_base) * 100
print("A acurácia do algoritmo de baseline foi %.2f%%" % acuracia)

data_x = teste_x[:,0]
data_y = teste_x[:,1]

x_min = data_x.min()
x_max = data_x.max()
y_min = data_y.min()
y_max = data_y.max()
print(x_min, x_max,y_min,y_max)

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min)/ pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min)/ pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]
Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)

import matplotlib.pyplot as plt

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(data_x, data_y, c=teste_y, s=1)
plt.show(block=False)
plt.pause(100)

