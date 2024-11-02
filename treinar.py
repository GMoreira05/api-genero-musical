import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import nltk
from nltk.corpus import stopwords

def treinar():
    try:
        nltk.download("stopwords")
        list_stop_words = stopwords.words("portuguese")

        # Carregar o dataset
        df = pd.read_csv("dataset_genero_musical.csv", delimiter=';', encoding='utf-8')

        df = df.dropna(subset=['letra'])
        df = df.drop_duplicates(subset=['letra'], keep='last')
        df = df.groupby('genero').sample(n=1602, random_state=42)
        df['letra'] = df['letra'].str.lower()

        # Separar os dados em features (letras) e target (gêneros)
        X = df['letra']
        y = df['genero']

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)

        # Criar o pipeline: vetorização das letras e o modelo Naïve Bayes
        model = Pipeline(
            [
                ('vect', CountVectorizer(stop_words= list_stop_words)), # Primeira Etapa
                ('clf', MultinomialNB()), # Segunda Etapa - Classificador
            ]
        )

        # Treinar o modelo
        model.fit(X_train.values, y_train.values)

        # Salvar o modelo treinado
        joblib.dump(model, 'modelo_musica.joblib')
    except Exception as e:
        print(e)

    print("Modelo treinado e salvo!")


if __name__ == "__main__":
    treinar()
