import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
import numpy as np
import re
import spacy
import nltk

import sys
sys.path.append('../..')

nltk.download('stopwords', quiet=True)
nltk.download('rslp', quiet=True) # Removedor de Sufixos da Língua Portuguesa

from nltk.tokenize import sent_tokenize # divide o comentário em frases
from unidecode import unidecode # remove acentos


# Importar variáveis das configurações (config.py)

from modules import config

ignorar = config.ignorar
dictionary = config.dictionary # substituições por regex para simplificar o léxico
remove_from_stopword_list = config.remove_from_stopword_list # remover da lista de stopwords
add_to_stopword_list = config.add_to_stopword_list
useless_terms = config.useless_terms # termos removidos por regex
blacklist = config.blacklist # frases com essas palavras são desconsideradas
lower_limit = config.lower_limit
upper_limit = config.upper_limit # máximo de palavras

# Função principal

def text_cleaner(df, n_processors=8, inference='false', remove_mandatory=True, stemming=True, social_network=None):
    '''Recebe o dataframe dos dados e trata a coluna "text", retornando um dataframe tratado e outro com os comentários removidos do tratamento.'''

    print('\nIniciando limpeza de texto')

    df = df.copy(deep=True)
    print('\nOriginais')
    print(df.text.reset_index(drop=True).to_string())

    print('\nRemover vazios')
    df = df.dropna(subset=['text']) # alguns comentários vêm vazios
    print(df.text.reset_index(drop=True).to_string())


    # Remover duplicados
    print('\nRemover duplicados')
    df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
    print(df.text.reset_index(drop=True).to_string())
    # Remover comentários que iniciam com @, mas que não são @loja
    df = df.loc[(~df.text.str.contains(r'^@', regex=True, na=False)) | (df.text.str.contains('@loja', regex=True, na=False))]
    print('\nRemover comentários que iniciam com @, mas que não são @loja')
    print(df.text.reset_index(drop=True).to_string())

    # Remover comentários que iniciam com nomes (duas palavras capitalizadas) mas mantê-los se contém loja
    df = df.loc[(~df.text.str.contains(r'^[A-ZÀ-Ú][a-zà-ú]+(\s|,)[A-ZÀ-Ú][a-zà-ú]{1,15}', regex=True, na=False)) | (df.text.str.contains('loja'))]
    print('\nRemover comentários que iniciam com nomes (duas palavras capitalizadas) mas mantê-los se contém loja')
    print(df.text.reset_index(drop=True).to_string())

    # ---------------------------------- modificações de texto


    # Lower case
    df.text = df.text.apply(str.lower)
    print('\nTransformar em caixa baixa')
    print(df.text.reset_index(drop=True).to_string())

    # Remover usernames
    df.text = df.text.apply(lambda x: re.sub('@[^\s]+', '', x))
    print('\nRemover usernames')
    print(df.text.reset_index(drop=True).to_string())

    # Remover "loja " do início de frases
    df.text = df.text.apply(lambda x: re.sub('^loja', '', x))
    print('\nRemover "loja" do início de frases')
    print(df.text.reset_index(drop=True).to_string())

    # Tokenizar os comentários, dividindo frases por pontos
    df.text = df.text.str.replace(r'\.\.+', '.', regex=True) # substituir "mais de 1 ponto" por 1 só
    df.text = df.text.str.replace(r'\s([?.!"](?:\s|$))', r'\1', regex=True) # remover espaços antes de pontuação
    df.text = df.text.str.replace(',', '.') # trocar vírgula por ponto ------------------- teste
    nltk.download('punkt', quiet=True) # lista de pontos para dividir frases
    df['tokenized'] = df.text.apply(sent_tokenize)


    # Expandir dataframe, onde cada frase do comentário se torna uma linha 
    df = df.explode(column='tokenized').reset_index(drop=True)
    df.text = df.tokenized.astype(str)
    df.drop(columns='tokenized', inplace=True)
    print('\nDividir comentários em frases')
    print(df.text.reset_index(drop=True).to_string())

    # Stopwords
    stop_words = nltk.corpus.stopwords.words('portuguese')

    for i in remove_from_stopword_list:
        stop_words.remove(i)

    removed = ', '.join(remove_from_stopword_list)

    for i in add_to_stopword_list:
        stop_words.append(i)

    added = ', '.join(add_to_stopword_list)

    def stopword_remover(text):
        return ' '.join([word for word in re.split('\W+', text) if word not in stop_words and word.isnumeric() is False])

    df.text = df.text.apply(stopword_remover)
    print('\nRemover stopwords')
    print(df.text.reset_index(drop=True).to_string())


    # Duplicated spaces and special characters 
    df.text = df.text.str.replace(r' +', ' ', regex=True) # remover excesso de espaços em branco
    df.text = df.text.str.replace(r'^\s+', '', regex=True) # remover espaços em branco do início de frases
    df.text = df.text.str.replace(r'kk+', '', regex=True) # remover kkkk
    df.text = df.text.str.replace(r'r\$', '', regex=True) # remover "R$"
    df.text = df.text.str.replace(r'(?:\\n)+', '. ', regex=True) # resquícios HTML
    df.text = df.text.str.replace(r'(?:\\t)+', '. ', regex=True) # quebras de texto são substituídas por pontos
    print('\nRemover excesso de espaços em branco, repetições de caracteres e resquícios HTML')
    print(df.text.reset_index(drop=True).to_string())

    df.text = df.text.str.replace(r'[^a-zA-ZÀ-ÿ ]+', '', regex=True) # remove what's left (not a letter or white space)
    print('\nRemover caracteres que não são letras ou espaço em branco')
    print(df.text.reset_index(drop=True).to_string())

    # Outliers de comprimento: textos muito grandes ou muito pequenos prejudicam o desempenho do algoritmo (ver config.py)

    df['text_size'] = df.text.str.split().apply(len)
    len_df = df.shape[0] # total sentences before filtering

    df.loc[df['text_size'] < lower_limit, 'str_len'] = 'small'
    df.loc[df['text_size'].between(lower_limit, upper_limit), 'str_len'] = 'medium'
    df.loc[df['text_size'] > upper_limit, 'str_len'] = 'large'

    # Com base no tamanho das frases, é utilizado o tamanho médio  

    df = df.loc[df.str_len == 'medium']
    print('\nRemover comentários muito grandes ou pequenos')
    print(df.text.reset_index(drop=True).to_string())


    # Lematização
    print('\nLematização')

    lemma_total = len(set(' '.join(df.text).split()))

    nlp = spacy.load('pt_core_news_md') # modelo que classifica part of speech
    nlp.max_length = 5000000

    df = df.reset_index(drop=True) # obrigatório para os índices se manterem corretos na criação da Series ao final, pois transformaremos df.text em lista
    
    docs = df.text.to_list()

    lemmatized_docs = []
    if inference == 'true': # a inferência automática na AWS não funcionou com processamento paralelo
        pipe = nlp.pipe(docs, disable=["parser", "ner"])
    else:
        pipe = nlp.pipe(docs, batch_size=n_processors, n_process=n_processors, disable=["parser", "ner"]) # melhora o desempenho do treino, mas não funciona na inferência

    for doc in pipe:
        sentece = []
        for word in doc:
            if ((word.pos_ == 'VERB') or (word.pos_ == 'ADJ') or (word.pos_ == 'NOUN')) and word.orth_ not in ignorar: # (config.py > ignorar)
                sentece.append(word.lemma_) # lemma
            else:
                sentece.append(word.orth_) # original
        sent = ' '.join(sentece)

        lemmatized_docs.append(sent)

    df.text = pd.Series(lemmatized_docs) # back to Series
    print(df.text.reset_index(drop=True).to_string())


    # Unidecode: esta etapa é aplicada após a lematização e antes da exclusão de palavras obrigatórias pois a lista 'mandatory' é de palavras lematizadas e sem acentos.
    df.text = df.text.apply(unidecode)
    print('\nSubstituição de letras com acento')
    print(df.text.reset_index(drop=True).to_string())

    # Substituições (pós-lematização) de alguns termos para melhorar o resultado (ver config.py > dictionary)
    print('\nRealizando substituições de sinônimos')
    def subs(text):
        for key in dictionary.keys():
            text = text.replace(key, dictionary[key])

        return text

    df.text = df.text.apply(subs)
    print(df.text.reset_index(drop=True).to_string())

    # Remover frases que não contenham as palavras obrigatórias
    if remove_mandatory:
        mandatory = config.mandatory
        contains_mandatory = df.text.str.contains('|'.join(mandatory))
        contains_blacklist = df.text.str.contains('|'.join(blacklist)) 

        df = df.loc[contains_mandatory & ~contains_blacklist] # remover frases contendo termos da blacklist e não contendo termos obrigatórios
        df = df.reset_index(drop=True)
    print('\nRemover frases que não contenham as palavras obrigatórias')
    print(df.text.reset_index(drop=True).to_string())

    # Termos inúteis (não utilizar antes da remoção de stopwords)
    def useless_terms_remover(text): # operação lenta para lista extensa de termos (ver config.py > useless_terms)

        for term in useless_terms:
            text = re.sub(r'{}'.format(term), ' ', text)

        return text

    df.text = df.text.apply(useless_terms_remover)
    print('\nRemover termos inúteis')
    print(df.text.reset_index(drop=True).to_string())

    # Stemming
    if stemming:

        stem_total = len(set(' '.join(df.text).split()))

        df = df.reset_index(drop=True)

        docs = df.text.to_list()
        stemmer = nltk.stem.RSLPStemmer()
        stemmed_docs = []
        for doc in docs:
            new_docs = []
            for word in doc.split():
                new_docs.append(stemmer.stem(word))
            stemmed_docs.append(' '.join(new_docs))

        df.text = pd.Series(stemmed_docs) # back to series
    print('\nStemização')
    print(df.text.reset_index(drop=True).to_string())


    df.text =  df.text.str.replace(r' +', ' ', regex=True) # remover excesso de espaços em branco
    df = df.loc[(df.text != '') & (df.text != ' ')] # remover textos vazios


    df = df.dropna(subset=['text']) # remover nulos
    print('\nRemover excesso de espaço em branco, vazios e nulos')
    print(df.text.reset_index(drop=True).to_string())
    

    return None