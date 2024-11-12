import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import nltk
from nltk.corpus import stopwords

def treinar():
    try:
        nltk.download("stopwords")
        list_stop_words = stopwords.words("portuguese")

        # Carregar o dataset
        df = pd.read_csv("dataset_genero_musical.csv", delimiter=';', encoding='utf-8')

        # Remover entradas com NaN na letra e duplicadas
        df = df.dropna(subset=['letra'])
        df = df.drop_duplicates(subset=['letra'], keep='last')

        # Obter o número mínimo de músicas por gênero
        min_musicas = df['genero'].value_counts().min()

        # Amostra igualitária: manter o número mínimo de músicas por gênero
        df = df.groupby('genero').sample(n=min_musicas, random_state=42)
        df['letra'] = df['letra'].str.lower()

        # Separar os dados em features (letras) e target (gêneros)
        X = df['letra']
        y = df['genero']

        # Criar o pipeline: vetorização das letras e o modelo Naïve Bayes
        model = Pipeline(
            [
                ('vect', CountVectorizer(stop_words= list_stop_words)), # Primeira Etapa
                ('clf', MultinomialNB()), # Segunda Etapa - Classificador
            ]
        )

        # Treinar o modelo com todo o dataset
        model.fit(X.values, y.values)

        # Salvar o modelo treinado
        joblib.dump(model, 'modelo_musica.joblib')
    except Exception as e:
        print(e)

    print("Modelo treinado e salvo!")

if __name__ == "__main__":
    treinar()
