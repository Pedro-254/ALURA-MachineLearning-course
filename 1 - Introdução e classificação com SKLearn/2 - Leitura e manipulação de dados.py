import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dados = pd.read_csv(uri)
# print(dados.head()) -> Printar os 5 primeiros dados do arquivo
mapa = {
    "home" : "principal",
    "how_it_works" : "como_funciona",
    "contact" : "contato",
    "bought" : "comprou",
}

dados = dados.rename(columns = mapa)

x = dados[["principal","como_funciona","contato"]]
y = dados["comprou"]

# Para nao enviesar os dados é sempre importante separar o treino do teste

# Determinar o tamanho dos dados
print("Quantidade de dados: Linhas: %d Colunas: %d "% (dados.shape[0], dados.shape[1]))

treino_x = x[:75]
treino_y = y[:75]

teste_x = x[75:]
teste_y = y[75:]

print("Treinaremos com %d elementos e testaremos com %d elementos." % (treino_x.shape[0],teste_x.shape[0]))

# Import do algoritmo
from sklearn.svm import LinearSVC
# Instanciando estimador
modelo = LinearSVC()
# Estimador aprende apartir de dados supervisionados.
modelo.fit(treino_x,treino_y)

# Calcula das previsoes
previsoes = modelo.predict(teste_x)

# Comparação com as respostas e previsoes
corretos = (previsoes == teste_y).sum()
print(corretos)

from sklearn.metrics import accuracy_score

# Calculo da acuracia feito de forma manual e com a função do SKlearn
acuracia = (corretos/len(teste_y))*100
acuracia = accuracy_score(teste_y, previsoes)*100
print("Acuracia: %.2f" % (acuracia))

print("\n------------------------- Separação com SKLEARN --------------------------")

# ----------------------------- SEPARAÇÃO TREINO E TESTE------------------------

# Como é algo basico em machine learning o proprio sklearn ja tem metodos para
# facilitar a separação entre treino e teste de dados.

from sklearn.model_selection import train_test_split

# >>>>>>>>>>>>>>>> SEDD >>>>>>>>>>>>>>>>
# Como a separação desse método por padrao é aleatoria, tem que ser criado uma seed
# para não ocorrer essa aleatorização.
SEED = 20
treino_x, teste_x, treino_y, teste_y = train_test_split(x,y,random_state=SEED,stratify=y, test_size= 0.25)

# >>>>>>>>>>>>>>>> STRATIFY >>>>>>>>>>>>>>>>
# Outro passo importante é a stratificação, pois é necessario que o treino tenha uma mesma
# proporção do teste, caso isso não ocorra pode acabar tendo divergencias.

print("---------- Conferindo a proporção da stratificação ----------")
print(treino_y.value_counts())
print(teste_y.value_counts())

print("Treinaremos com %d elementos e testaremos com %d elementos." % (treino_x.shape[0],teste_x.shape[0]))

# Import do algoritmo
from sklearn.svm import LinearSVC
# Instanciando estimador
modelo = LinearSVC()
# Estimador aprende apartir de dados supervisionados.
modelo.fit(treino_x,treino_y)

# Calcula das previsoes
previsoes = modelo.predict(teste_x)

# Comparação com as respostas e previsoes
corretos = (previsoes == teste_y).sum()
print(corretos)

from sklearn.metrics import accuracy_score

# Calculo da acuracia feito de forma manual e com a função do SKlearn
acuracia = (corretos/len(teste_y))*100
acuracia = accuracy_score(teste_y, previsoes)*100
print("Acuracia: %.2f" % (acuracia))

