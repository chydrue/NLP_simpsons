from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

pd.set_option('display.max_colwidth', 150)


df_treino = pd.read_csv('simpsons_train.csv')
df_valida = pd.read_csv ('simpsons_test.csv')
print(df_treino.columns)

treino_x, treino_y = df_treino['spoken_words'].str.rsplit(pat='-').str[-1], df_treino['character_name']
valida_x, valida_y = df_valida['spoken_words'].str.rsplit(pat='-').str[-1], df_valida['character_name']

model = MultinomialNB()
print('Precisão para diferentes métodos de vetorização:\n')

vec_count = CountVectorizer(stop_words='english')
treinoX_count = vec_count.fit_transform(treino_x).toarray()
validaX_count = vec_count.transform(valida_x).toarray()
model.fit(treinoX_count, treino_y)
eficiencia_count = model.score(validaX_count, valida_y) * 100
print(f'CountVectorizer: {eficiencia_count}%')

vec_tfid1 = TfidfVectorizer(analyzer='word')
treinoX_tfidf = vec_tfid1.fit_transform(treino_x).toarray()
validaX_tfidf = vec_tfid1.transform(valida_x).toarray()
model.fit(treinoX_tfidf, treino_y)
eficiencia_tfidf = model.score(validaX_tfidf, valida_y) * 100
print(f'TF-IDF [Utilizando palavras]: {eficiencia_tfidf}%')

vec_ngram = TfidfVectorizer(analyzer='word', ngram_range=(2,3), max_features=10000)
treinoX_ngram = vec_ngram.fit_transform(treino_x)
validaX_ngram = vec_ngram.transform(valida_x)
model.fit(treinoX_ngram, treino_y)
eficiencia_ngram = model.score(validaX_ngram, valida_y) * 100
print(f'TF-IDF [Utilizando n-gram(2,3)]: {eficiencia_ngram}%')

vec_char = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=10000)
treinoX_char = vec_char.fit_transform(treino_x)
validaX_char = vec_char.transform(valida_x)
model.fit(treinoX_char, treino_y)
eficiencia_char = model.score(validaX_char, valida_y) * 100
print(f'TF-IDF [Utilizando letras]: {eficiencia_char}%')






