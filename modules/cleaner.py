import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
import numpy as np
import re
import spacy
import nltk

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

def text_cleaner(df, n_processors=8, inference='false', remove_mandatory=True, stemming=True, social_network=None):
    '''Recebe o dataframe dos dados e trata a coluna "text", retornando um dataframe tratado e outro com os comentários removidos do tratamento.'''

    df = df.copy(deep=True)

    print('Iniciando limpeza de texto')
    print('Removendo vazios')
    df = df.dropna(subset=['text']) # alguns comentários vêm vazios


    # Remover duplicados
    print('Proporção de duplicados removidos:', (100 * df.duplicated(subset=['text', 'user_name']).sum()/df.duplicated(subset=['text', 'user_name']).count()).round(2), '%')
    df = df.drop_duplicates(subset=['text', 'user_name'], keep='first').reset_index(drop=True)


    # Remover da url o código do comentário, para manter somente a url da postagem
    df.url = df.url.fillna('no_url')

    if social_network == 'instagram':
        df.url = df.url.apply(lambda x: '/'.join(filter(None, x.split('/')[:-1])).replace(':/', '://'))

    if social_network == 'facebook':
        df.url = df.url.apply(lambda x: '/'.join(filter(None, x.split('?')[:-1])))


    # Coletar data da postagem (data do primeiro comentário capturado pela HiPlatform)
    df.interaction_date = pd.to_datetime(df.interaction_date)
    df = pd.merge(df, df.groupby('url', as_index=False).agg(publish_date=('interaction_date', 'min')), how='left')


    # Salvar original
    print('Salvando texto original')
    df['original_text'] = df.text


    # Remover comentários que iniciam com @, mas que não são @loja
    print('Removendo comentários que iniciam com @, mas que não são @loja')
    df_ = df.loc[(~df.original_text.str.contains(r'^@', regex=True, na=False)) | (df.original_text.str.contains('@loja', regex=True, na=False))]
    df_holdout = df.loc[~df.index.isin(df_.index)]
    df = df_


    # Remover comentários que iniciam com nomes (duas palavras capitalizadas) mas mantê-los se contém loja
    print('Removendo comentários que iniciam com nomes (duas palavras capitalizadas) mas mantendo-os se contém loja')
    df_ = df.loc[(~df.original_text.str.contains(r'^[A-ZÀ-Ú][a-zà-ú]+(\s|,)[A-ZÀ-Ú][a-zà-ú]{1,15}', regex=True, na=False)) | (df.original_text.str.contains('loja'))]
    df_holdout_1 = df.loc[~df.index.isin(df_.index)]
    df = df_
    # df_holdout_1 = pd.DataFrame()

    # ---------------------------------- modificações de texto


    # Lower case
    print('Transformar em caixa baixa')
    df.text = df.text.apply(str.lower)


    # Remover usernames
    print('Remover usernames')
    df.text = df.text.apply(lambda x: re.sub('@[^\s]+', '', x))


    # Remover "loja " do início de frases
    print('Remover "loja" do início de frases')
    df.text = df.text.apply(lambda x: re.sub('^loja', '', x))

    print('Dividindo comentários em frases')
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


    # Stopwords
    print('Removendo stopwords')
    stop_words = nltk.corpus.stopwords.words('portuguese')

    for i in remove_from_stopword_list:
        stop_words.remove(i)

    removed = ', '.join(remove_from_stopword_list)
    print(f'Removidas palavras da lista de stopwords: {removed}')

    for i in add_to_stopword_list:
        stop_words.append(i)

    added = ', '.join(add_to_stopword_list)
    print(f'Adicionadas palavras na lista de stopwords: {added}')

    def stopword_remover(text):
        return ' '.join([word for word in re.split('\W+', text) if word not in stop_words and word.isnumeric() is False])

    df.text = df.text.apply(stopword_remover)
    

    # Duplicated spaces and special characters 
    print('Removendo excesso de espaços em branco, letras repetidas, símbolos e resquícios HTML')
    df.text = df.text.str.replace(r' +', ' ', regex=True) # remover excesso de espaços em branco
    df.text = df.text.str.replace(r'^\s+', '', regex=True) # remover espaços em branco do início de frases
    df.text = df.text.str.replace(r'kk+', '', regex=True) # remover kkkk
    df.text = df.text.str.replace(r'r\$', '', regex=True) # remover "R$"
    df.text = df.text.str.replace(r'(?:\\n)+', '. ', regex=True) # resquícios HTML
    df.text = df.text.str.replace(r'(?:\\t)+', '. ', regex=True) # quebras de texto são substituídas por pontos

    print('Removendo o que não é letra ou espaço em branco')
    df.text = df.text.str.replace(r'[^a-zA-ZÀ-ÿ ]+', '', regex=True) # remove what's left (not a letter or white space)
    

    # Outliers de comprimento: textos muito grandes ou muito pequenos prejudicam o desempenho do algoritmo (ver config.py)
    print('Removendo comentários muito grandes ou pequenos')
    df['text_size'] = df.text.str.split().apply(len)
    len_df = df.shape[0] # total sentences before filtering

    df.loc[df['text_size'] < lower_limit, 'str_len'] = 'small'
    df.loc[df['text_size'].between(lower_limit, upper_limit), 'str_len'] = 'medium'
    df.loc[df['text_size'] > upper_limit, 'str_len'] = 'large'
    
    size_upper = df.loc[df.str_len == 'large'].shape[0]
    size_lower = df.loc[df.str_len == 'small'].shape[0]

    print(f'{size_lower} frases com menos de {lower_limit} palavras foram removidas.')
    print(f'{size_upper} frases com mais de {upper_limit} palavras foram removidas.')

    # Com base no tamanho das frases, é utilizado o tamanho médio  

    df = df.loc[df.str_len == 'medium']
    df_ = df.drop(columns=['text_size', 'str_len'])
    print(f'Restaram {df_.shape[0]} frases de um total de {len_df}.')

    df_holdout_5 = df.loc[~df.index.isin(df_.index)].drop(columns=['text_size', 'str_len']) # salvar frases que não serão consideradas para retornar ao final
    df = df_.reset_index(drop=True)


    # Lematização
    print('Lematizando')

    lemma_total = len(set(' '.join(df.text.tolist()).split()))
    print(f'Total de palavras únicas antes da lematização: {lemma_total}')

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

    lemma_total = len(set(' '.join(df.text.tolist()).split()))
    print(f'Total de palavras únicas após lematização: {lemma_total}')


    # Unidecode: esta etapa é aplicada após a lematização e antes da exclusão de palavras obrigatórias pois a lista 'mandatory' é de palavras lematizadas e sem acentos.
    print('Substituição de letras com acento')
    df.text = df.text.apply(unidecode)


    # Substituições (pós-lematização) de alguns termos para melhorar o resultado (ver config.py > dictionary)
    print('Realizando substituições de sinônimos')
    def subs(text):
        for key in dictionary.keys():
            text = text.replace(key, dictionary[key])

        return text

    df.text = df.text.apply(subs)


    # Remover frases que não contenham as palavras obrigatórias
    print('Removendo frases que não contenham as palavras obrigatórias')
    if remove_mandatory:
        mandatory = config.mandatory
        contains_mandatory = df.text.str.contains('|'.join(mandatory))
        contains_blacklist = df.text.str.contains('|'.join(blacklist)) 

        df_ = df.loc[contains_mandatory & ~contains_blacklist] # remover frases contendo termos da blacklist e não contendo termos obrigatórios
        df_holdout_2 = df.loc[~df.index.isin(df_.index)] # salvar dataframe de frases excluídas para concatenar ao final
        df = df_.reset_index(drop=True)
        total_removed = round(100 * df_holdout_2.shape[0]/(df_holdout_2.shape[0] + df.shape[0]), 1)

        print(f'{total_removed}% das frases foram removidas.')


    # Termos inúteis (não utilizar antes da remoção de stopwords)
    print('Removendo termos inúteis')
    def useless_terms_remover(text): # operação lenta para lista extensa de termos (ver config.py > useless_terms)

        for term in useless_terms:
            text = re.sub(r'{}'.format(term), ' ', text)

        return text

    df.text = df.text.apply(useless_terms_remover)


    # Stemming
    if stemming:
        print('Stemizando')

        stem_total = len(set(' '.join(df.text.tolist()).split()))
        print(f'Total de palavras únicas antes do stemming: {stem_total}')

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

        stem_total = len(set(' '.join(df.text.tolist()).split()))
        print(f'Total de palavras únicas após stemming: {stem_total}')


    print('Removendo excesso de espaço em branco, vazios e nulos')
    df.text =  df.text.str.replace(r' +', ' ', regex=True) # remover excesso de espaços em branco
    df_ = df.loc[(df.text != '') & (df.text != ' ')] # remover textos vazios
    df_holdout_4 = df.loc[~df.index.isin(df_.index)]
    df = df_.reset_index(drop=True)


    df_ = df.dropna(subset=['text']) # remover nulos
    df_holdout_3 = df.loc[~df.index.isin(df_.index)]
    df = df_.reset_index(drop=True)

    print('Pré-processamento concluído')
    

    # Concatenar comentários removidos
    df_removed = pd.concat([df_holdout, df_holdout_1, df_holdout_2, df_holdout_3, df_holdout_4, df_holdout_5], ignore_index=True)


    return df, df_removed