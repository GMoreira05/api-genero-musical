from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import scrap
import treinar
import pandas as pd

# Inicializar o FastAPI
app = FastAPI()

# Definir o modelo de dados que a API vai receber
class Musica(BaseModel):
    letra: str

# Rota principal para prever o gênero
@app.post("/prever_genero/")
async def prever_genero(musica: Musica):
    # Carregar o modelo treinado
    modelo = joblib.load('modelo_musica.joblib')
    # Fazer a previsão do gênero
    genero_previsto = modelo.predict([musica.letra])[0]
    print(modelo.predict([musica.letra]))
    return {"genero": genero_previsto}

# Rota simples para verificar se a API está rodando
@app.get("/")
async def root():
    return {"message": "API de previsão de gênero musical está funcionando!"}

# Rota para inserir mais músicas no dataset
@app.get("/dumpar_artista/")
async def dumpar_artista(artista):
    try:
        musicas = scrap.get_links(artista)
        scrap.salvar_arquivo(musicas)
        treinar.treinar()
        return {"message": "Músicas inseridas com sucesso"}
    except Exception as e:
        return {"erro": "Não foi possível acessar a página do artista. Insira de acordo com a URL do letras.mus.br"}

# Rota 1: Contar quantidade de músicas por gênero
@app.get("/contar_generos/")
async def contar_generos():
    df = pd.read_csv('dataset_genero_musical.csv', delimiter=';')
    df = df.drop_duplicates(subset=['letra'], keep='last')
    contagem = df['genero'].value_counts().to_dict()
    return {"contagem_de_generos": contagem}

# Rota 2: Listar todos os autores únicos
@app.get("/listar_autores/")
async def listar_autores():
    df = pd.read_csv('dataset_genero_musical.csv', delimiter=';')
    df = df.drop_duplicates(subset=['letra'], keep='last')
    autores = df['autor'].unique().tolist()
    return {"autores": autores}

# Rota 3: Top 5 autores com mais músicas
@app.get("/top5_autores/")
async def top5_autores():
    df = pd.read_csv('dataset_genero_musical.csv', delimiter=';')
    df = df.drop_duplicates(subset=['letra'], keep='last')
    top_autores = df['autor'].value_counts().head(5).to_dict()
    return {"top5_autores": top_autores}

# Rota 4: Contagem de músicas por autor
@app.get("/contar_musicas_por_autor/")
async def contar_musicas_por_autor():
    df = pd.read_csv('dataset_genero_musical.csv', delimiter=';')
    df = df.drop_duplicates(subset=['letra'], keep='last')
    contagem = df['autor'].value_counts().to_dict()
    return {"contagem_musicas_por_autor": contagem}

# Rota 5: Tamanho médio das letras de música por gênero
@app.get("/tamanho_medio_letras_por_genero/")
async def tamanho_medio_letras_por_genero():
    df = pd.read_csv('dataset_genero_musical.csv', delimiter=';')
    df = df.drop_duplicates(subset=['letra'], keep='last')
    df['tamanho_letra'] = df['letra'].apply(len)
    tamanho_medio = df.groupby('genero')['tamanho_letra'].mean().to_dict()
    return {"tamanho_medio_letras_por_genero": tamanho_medio}

# Rota 6: Top 5 autores de cada gênero com mais músicas
@app.get("/top5_autores_por_genero/")
async def top5_autores_por_genero():
    df = pd.read_csv('dataset_genero_musical.csv', delimiter=';')
    df = df.drop_duplicates(subset=['letra'], keep='last')
    
    top_autores_por_genero = {}
    
    # Agrupa o dataset por gênero e, para cada gênero, obtém os 5 autores com mais músicas
    for genero, grupo in df.groupby('genero'):
        top_autores = grupo['autor'].value_counts().head(5).to_dict()
        top_autores_por_genero[genero] = top_autores

    return {"top5_autores_por_genero": top_autores_por_genero}