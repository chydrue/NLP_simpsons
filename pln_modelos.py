from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, CategoricalNB, GaussianNB, ComplementNB
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_colwidth', 150)


df_treino = pd.read_csv('simpsons_train.csv')
df_valida = pd.read_csv ('simpsons_test.csv')
print(df_treino.columns)


treino_x, treino_y = df_treino['spoken_words'].str.rsplit(pat='-').str[-1], df_treino['character_name']
valida_x, valida_y = df_valida['spoken_words'].str.rsplit(pat='-').str[-1], df_valida['character_name']

vec = CountVectorizer(stop_words='english')
treino_x = vec.fit_transform(treino_x).toarray()
valida_x = vec.transform(valida_x).toarray()

print("Precisão para diferentes modelos de classificadores Naive-Bayes:\n")

model_multinomial = MultinomialNB()
model_multinomial.fit(treino_x, treino_y)
eficiencia_multinomial = model_multinomial.score(valida_x, valida_y) * 100
print(f"Multinomial: {eficiencia_multinomial}%")

model_bernoulli = BernoulliNB()
model_bernoulli.fit(treino_x, treino_y)
eficiencia_bernoulli = model_bernoulli.score(valida_x, valida_y) * 100
print(f"Bernoulli: {eficiencia_bernoulli}%")

model_complement = ComplementNB()
model_complement.fit(treino_x, treino_y)
eficiencia_complement = model_complement.score(valida_x, valida_y) * 100
print(f"Complement: {eficiencia_complement}%")

model_gaussian = GaussianNB()
model_gaussian.fit(treino_x, treino_y)
eficiencia_gaussian = model_gaussian.score(valida_x, valida_y) * 100
print(f"Gaussian: {eficiencia_gaussian}%")

model_categorical = CategoricalNB()
model_categorical.fit(treino_x, treino_y)
eficiencia_categorical = model_categorical.score(valida_x, valida_y) * 100
print(f"Categorical: {eficiencia_categorical}%")




