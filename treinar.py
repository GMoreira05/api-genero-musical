import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

def treinar():
    try:
        # Carregar o dataset
        df = pd.read_csv("dataset_genero_musical.csv", delimiter=';', encoding='utf-8')

        df = df.dropna(subset=['letra'])
        df = df.drop_duplicates(subset=['letra'], keep='last')

        # Separar os dados em features (letras) e target (gêneros)
        X = df['letra']
        y = df['genero']

        # Criar o pipeline: vetorização das letras e o modelo Naïve Bayes
        model = make_pipeline(TfidfVectorizer(), MultinomialNB())

        # Treinar o modelo
        model.fit(X, y)

        # Salvar o modelo treinado
        joblib.dump(model, 'modelo_musica.joblib')
    except Exception as e:
        print(e)

    print("Modelo treinado e salvo!")


if __name__ == "__main__":
    treinar()
