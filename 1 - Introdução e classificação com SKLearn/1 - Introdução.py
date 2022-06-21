# Estimador basico de classificação
from sklearn.svm import LinearSVC

# Caracteristicas do animal (1 sim, 0 não)
# Pelo longo?
# Perna curta?
# Faz auau?
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1 ,1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

#  1 => porco, 0 => cachorro
dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
classes = [1,1,1,0,0,0]

# Apartir desses dados previamente selecionados e classes previamente estipuladas, precisariamos de algum
# estimador para poder classificar dados novos.

# Instanciando estimador
model = LinearSVC()

# Estimador aprende apartir de dados supervisionados.
model.fit(dados,classes)

# Apartir dos dados fornecidos o algoritmo de ML consegue classificar um outro animal misterioso.
animal_misterioso = [1,1,1]
print(model.predict([animal_misterioso]))

misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]

testes = [misterio1,misterio2,misterio3]
print("Predição dos animais misteriosos:")
previsoes = model.predict(testes)
print(previsoes)

# Apesar de ter passados esses animais como um misterio para o algoritmo, eu ja tinha essa resposta
# Resp: Cachorro, Porco, Porco.
# O que demonstra um erro no algoritmo que teve como resustado: [0,1,0].

testes_classes = [0,1,1]

# Podemos comparar os resultados do algoritmo e da responsta.
corretos = ((previsoes == testes_classes).sum())
total = len(testes)

# Assim podendo fazer uma estimativa da taxa de acerto do algoritmo!
taxa_de_acerto = corretos/total
print("Taxa de acerto: %.2f" % (taxa_de_acerto*100))

# Apesar de que o sklearn ja tem uma função que mede a taxa de acerto de um algoritmo.
from sklearn.metrics import accuracy_score
# accuracy_score([valores verdadeiros], [valores de teste])
taxa_de_acerto = accuracy_score(testes_classes, previsoes)
print("(sklearn) Taxa de acerto: %.2f" % (taxa_de_acerto*100))