from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost, numpy, textblob, string
# from keras.preprocessing import text, sequence
# from keras import layers, models, optimizers

pd.set_option('display.max_colwidth', 150)

df_treino = pd.read_csv('simpsons_train.csv')
df_valida = pd.read_csv ('simpsons_test.csv')
print(df_treino.columns)

# treino_menor_x = df_valida['spoken_words']
# treino_menor_y = df_valida['character_name']
treino_menor_x, treino_menor_y = df_treino['spoken_words'].str.rsplit(pat='-').str[-1], df_treino['character_name']
valida_menor_x, valida_menor_y = df_valida['spoken_words'].str.rsplit(pat='-').str[-1], df_treino['character_name']

# df_treino['spoken_words'] = df_treino['spoken_words'].str.strip().str.lower()
# df_treino['character_name'] = df_treino['character_name'].str.strip().str.lower()
# df_valida['spoken_words'] = df_valida['spoken_words'].str.strip().str.lower()
# df_valida['character_name'] = df_valida['character_name'].str.strip().str.lower()

treino_menor_x, valida_menor_x, treino_menor_y, valida_menor_y = train_test_split(treino_menor_x, treino_menor_y, stratify=treino_menor_y, test_size=0.25, random_state=42)

# print(treino_menor_x.shape)
# print(treino_menor_y.shape)
# print(df_treino.isnull().sum(axis = 0))
# print(valida_menor_x.shape)
# print(valida_menor_y.shape)

# treino_x, treino_y = df_treino['spoken_words'].str.rsplit(pat='-').str[-1], df_treino['character_name']
# valida_x, valida_y = df_valida['spoken_words'].str.rsplit(pat='-').str[-1], df_treino['character_name']

# print(treino_x.shape[0])

# print(treino_x.sample(5))
# print(valida_x.sample(5))

vec = CountVectorizer(stop_words='english')
treino_menor_x = vec.fit_transform(treino_menor_x).toarray()
valida_menor_x = vec.transform(valida_menor_x).toarray()

# vec = CountVectorizer(stop_words='english')
# treino_x = vec.fit_transform(treino_x).toarray()
# valida_x = vec.transform(valida_x).toarray()

model = MultinomialNB()
model.fit(treino_menor_x, treino_menor_y)

# model = MultinomialNB()
# model.fit(treino_x, treino_y)
#
print(model.score(valida_menor_x, valida_menor_y))
# model.score(valida_x, valida_y)


# encoder = preprocessing.LabelEncoder()
# treino_y = encoder.fit_transform(treino_y)
# valida_y = encoder.fit_transform(valida_y)


# print(df_treino['spoken_words'].head(5))