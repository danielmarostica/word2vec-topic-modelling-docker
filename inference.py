# -*- coding: utf-8 -*-
'''
A inferência trará do Redshift Spectrum dados referentes a comentários nos últimos 7 dias, e retornará "weekly_output.csv" com as classificações dos mesmos.
'''

import pandas as pd
import tarfile
import joblib
import csv
import numpy as np
import sys
sys.path.append('modules')

import re
import boto3
import sagemaker

from modules.cleaner import text_cleaner
from modules import config

def connect_aws(social_network):
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    prefix = f'social-media-cx-monitor/loja_br/comentarios_{social_network}'

    return bucket, prefix

def retrieve_data(sequential: bool, bucket, prefix, social_network):
    '''Permite execução normal (últimos 30 dias) ou sequencial, para simular execuções em dias consecutivos (ver modules/inference_sequential).'''    

    # role = iam.get_role(RoleName='datascience-sagemaker-s3-redshift')['Role']['Arn']
    s3_client = boto3.client('s3')


    # load training job name
    client = boto3.client(service_name='sagemaker')
    training_job_name = client.list_training_jobs(NameContains=f'cx-social-comments-loja-br-{social_network}', SortBy='CreationTime', MaxResults=100)
    training_job_name = training_job_name['TrainingJobSummaries'][0]['TrainingJobName']


    if sequential:
        root = '../temp' # inference_sequential.py is inside the "extra" folder
    else:
        root = 'temp'


    # get sagemaker model attributes/weights
    s3_client.download_file(bucket, f'{prefix}/training-jobs/{training_job_name}/output/model.tar.gz', f'{root}/model.tar.gz')
    tar = tarfile.open(f'{root}/model.tar.gz')
    tar.extractall(f'{root}/')
    model = joblib.load(f'{root}/model.joblib')


    # get cluster names
    clusters = pd.read_json(f'{root}/clusters.json').reset_index().rename({'index': 'subclusters_id'}, axis=1)

    # get data for inference
    if sequential:
        original_data = pd.read_csv('data_for_inference.csv', delimiter=';', quotechar='"')
    else:
        original_data = pd.read_csv(f's3://{bucket}/{prefix}/data_for_inference.csv', delimiter=';', escapechar='\\')

    clean_data, removed = text_cleaner(original_data, inference=True, stemming=True, social_network=social_network)

    return model, clusters, clean_data, removed


def get_cluster_size(kmeans, clusters):

    cluster_size = pd.Series(kmeans.labels_, name='size').value_counts().sort_index()
    clusters = pd.concat([clusters, cluster_size], axis=1)

    return clusters


def infer(model, clean_data):

    # predict
    predictions = model.predict(clean_data.text)

    return predictions

def post_process(predictions, clusters, clean_data, removed):

    # Adicionar output do predict ao dataframe
    df_clustered = pd.concat([clean_data, pd.Series(predictions, name='subclusters_id')], axis=1)
    df_clustered = pd.merge(df_clustered, clusters)


    # Concatenar comentários que não foram tratados por não conterem palavras obrigatórias
    df_clustered = pd.concat([df_clustered, removed]).reset_index(drop=True)


    # Inserir sentimento do cluster
    df_clustered.loc[df_clustered.cluster.isin(config.positivos), 'sentimento'] = 'Positivo'
    df_clustered.loc[df_clustered.cluster.isin(config.negativos), 'sentimento'] = 'Negativo'
    df_clustered.loc[df_clustered.cluster.isin(config.outros), 'sentimento'] = 'Outro'
    df_clustered.loc[df_clustered.cluster == 'Cluster não identificado', 'sentimento'] = 'Não identificado'
    df_clustered.loc[df_clustered.cluster.isna(), 'sentimento'] = 'Não identificado'


    # Agrupar por sentimento e coletar a moda dos clusters
    def moda(df_in):
        mode = list(pd.Series.mode(df_in.cluster)) # compute mode

        if len(mode) > 1: # if no mode, choose the biggest cluster
            biggest_cluster = max(df_in['size'])
            biggest_cluster_name = df_in.loc[df_in['size'] == biggest_cluster]['cluster'].tolist()[0]
            return biggest_cluster_name
        elif mode == []:
            return None
        else:
            return mode[0]

    df_mode = df_clustered.groupby(['user_name', 'original_text', 'sentimento'])[['cluster', 'size']].apply(moda).reset_index().rename({0: 'cluster_mode'}, axis=1)


    # Substituir o cluster pela moda
    df_clustered = pd.merge(df_clustered, df_mode, how='left').drop('cluster', axis=1).rename({'cluster_mode': 'cluster'}, axis=1)


    # Criar sentimentos para sentimento, ordernar da menos para a mais importante e, por fim, dropar duplicados mantendo os mais importantes (problemas).
    df_clustered.sentimento = pd.Categorical(df_clustered.sentimento, categories=['Não identificado', 'Outro', 'Positivo', 'Negativo'])
    df_clustered = df_clustered.sort_values(['sentimento', 'size']).drop_duplicates(subset=['original_text'], keep='last')
    df_clustered.sentimento = df_clustered.sentimento.replace('Não identificado', np.nan)


    # A cluster mais incerta é ignorada
    df_clustered.cluster = df_clustered.cluster.replace('Cluster não identificado', np.nan)

    return df_clustered


def get_mode(df_clustered):
    # Coletar a moda do problema dos últimos 30 dias
    def get30daysMode(df_in):
        mode = df_in.loc[df_in.interaction_date.between(df_in.interaction_date.max() - pd.Timedelta('30 days'), df_in.interaction_date.max())].problema.mode().tolist()
        if mode != []:
            if len(mode) > 1:
                return df_in.sort_values('sentimento').drop_duplicates(subset=['user_name'], keep='last')['problema'].tolist()[0] # classificado por importância da sentimento
            else:
                return mode[0]
        else:
            return None

    df_clustered.interaction_date = pd.to_datetime(df_clustered.interaction_date) # garantir coerência
    problema_30d = pd.Series(df_clustered.groupby('user_name').apply(get30daysMode), name='problema_30d')


    df_clustered = pd.merge(df_clustered, problema_30d, left_on='user_name', right_index=True, how='left')

    return df_clustered


def prepare_for_upload(df_clustered):

    # Eliminar colunas desnecessárias
    df_clustered.drop(columns=['text', 'clusters', 'size'], inplace=True, errors='ignore')


    # Coletar números de pedidos
    df_clustered.original_text = df_clustered.original_text.fillna('') # re.findall não funciona com nans
    df_clustered['order_number'] = df_clustered.original_text.apply(lambda x: re.findall(r'(\d{8,10})', x) or None).str[0]


    df_clustered.interaction_date = pd.to_datetime(df_clustered.interaction_date).dt.date

    df_clustered = df_clustered[['interaction_date', 'user_name', 'url', 'content_type', 'original_text', 'cluster', 'sentimento', 'problema', 'problema_30d', 'order_number']] # ordenar

    return df_clustered


def upload(df_clustered):

    df_clustered.to_csv(f's3://{bucket}/{prefix}/weekly-outputs/weekly_output.csv', index=False, quoting=csv.QUOTE_ALL)

    print('Output uploaded to S3')


if __name__ == "__main__":

    social_network = sys.argv[1]

    bucket, prefix = connect_aws(social_network=social_network)

    model, clusters, clean_data, removed = retrieve_data(sequential=False, bucket=bucket, prefix=prefix, social_network=social_network)

    clusters = get_cluster_size(model['clusterer'], clusters)

    predictions = infer(model, clean_data)

    df_clustered = post_process(predictions, clusters, clean_data, removed)
    
    df_clustered = get_mode(df_clustered)
    
    df_clustered = prepare_for_upload(df_clustered)

    upload(df_clustered)