import string
import sys
from urllib.request import urlopen
import treinar

import nltk
from bs4 import BeautifulSoup, NavigableString

url = 'https://www.letras.mus.br'
music = ' '

def get_artistas(genero):
    url_genero = 'https://www.letras.mus.br/mais-acessadas/'
    artistas = []
    try:
        ctrl_page = set()
        html = urlopen(f"{url_genero}/{genero}")
        bs = BeautifulSoup(html, 'html.parser')
        
        for li in bs.find_all('ol', {'class': 'top-list_art'}):
            try:
                links = li.find_all('a')
                index = 0
                for link in links:
                    artistas.append(link.attrs['href'])
                    if(index==19):
                        break
                    index+=1
            except NavigableString: 
                pass
            
        return artistas
            
    except Exception as e:
        raise Exception(f'Ocorreu algum erro ao tentar acessar o site. {e}')
        
def get_links(band):
    try:        
        musicas = []
        
        ctrl_page = set()
        html = urlopen(f"{url}/{band}")
        bs = BeautifulSoup(html, 'html.parser')
        
        for table in bs.find_all('div', {'class': 'songList-table'}):
            for a in table.findChildren('a'):
                if 'href' in a.attrs:
                    if a.attrs['href'] not in ctrl_page:
                        ctrl_page.add(f"{a.attrs['href']}")
                        musica = get_music(a.attrs['href'])
                        musicas.append(musica)
                        
        return musicas
    except Exception as e:
        raise Exception(f'Ocorreu algum erro ao tentar acessar o site. {e}')


def get_music(new_page):
    global music
    try:
        html = ''
        for attempt in range(5):
            try:
                html = urlopen(f"{url}/{new_page}")
            except:
                print(f'Falha ao abrir {new_page}. Tentando novamente...')
            else:
                break
        else:
            raise Exception('Falhou todas as tentativas de abrir a página')
        
        bs = BeautifulSoup(html, 'html.parser')
        
        genero = bs.find('div', {'id': 'breadcrumb'}).find_all('span')[2].find('span').text.replace(';', '')
        titulo = bs.find('div', {'class': 'title-content'}).find('div').find('h1').text.strip().replace(';', '')
        autor = bs.find('div', {'class': 'title-content'}).find('div').find('a').find('h2').text.strip().replace(';', '')
                
        letra = ""
        for verse in bs.find('div', {'class': 'lyric-original'}).find_all('p'):
            letra += ' '.join(verse.stripped_strings)
            letra += ' '
        
        letra = letra.replace(';', '')
        
        return {'genero': genero, 'letra': letra, 'titulo': titulo, 'autor': autor}
    except Exception as e:
        print(f'Ocorreu algum erro ao tentar acessar o site. {e}')


def salvar_arquivo(musicas):
    print('Finalizando...')
    texto = ""
    for musica in musicas:
        texto+=f"\n{musica['titulo']};{musica['autor']};{musica['letra']};{musica['genero']}"
    try:
        with open("dataset_genero_musical.csv", "a", encoding="utf-8") as dataset_dumpado:
            dataset_dumpado.write(f"{texto}")
    except Exception as e:
        print(f'Ocorreu algum erro ao tentar gravar o arquivo. {e}')


if __name__ == "__main__":
    treinar.treinar()
