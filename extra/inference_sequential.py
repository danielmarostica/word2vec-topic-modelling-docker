# -*- coding: utf-8 -*-
'''
Utilizando o modelo mais recente hospedado no S3, faz inferência sequencial para a cada dia atualizar a moda dos últimos 30 dias de cada usuário.
'''

import sys
sys.path.append("..")

from datetime import datetime, timedelta
import pandas as pd
import csv
import sys
#sys.path.append('modules')

from inference import connect_aws, retrieve_data, get_cluster_size, infer, post_process, prepare_for_upload


start_date = datetime(2021, 4, 1) # data de início da simulação
end_date = datetime.today() - timedelta(1)


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


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def apply_get_mode(df_clustered, start_date, end_date):
    for execution_day in daterange(start_date, end_date):
        print(f'Execution date: {execution_day}')
        condition = df_clustered.interaction_date.between(execution_day - timedelta(days=30), execution_day)
        problema_30d = pd.Series(df_clustered.loc[condition].groupby('user_name').apply(get30daysMode), name='problema_30d')

        intermerge = pd.merge(df_clustered.loc[condition].drop('problema_30d', axis=1, errors='ignore'), problema_30d, left_on='user_name', right_index=True, how='left')

        df_clustered = pd.concat([df_clustered.loc[~condition], intermerge])

    return df_clustered


if __name__=='__main__':

    social_network = sys.argv[1]

    bucket, prefix = connect_aws(social_network)

    model, clusters, clean_data, removed = retrieve_data(sequential=True, bucket=bucket, prefix=prefix, social_network=social_network)

    clusters = get_cluster_size(model['clusterer'], clusters)

    predictions = infer(model, clean_data)

    df_clustered = post_process(predictions, clusters, clean_data, removed)

    # Loop para atualizar sequencialmente a moda de 30 dias
    df_clustered.interaction_date = pd.to_datetime(df_clustered.interaction_date) # garantir coerência

    df_clustered = apply_get_mode(df_clustered, start_date, end_date)

    df_clustered = df_clustered.sort_values('interaction_date').reset_index(drop=True)

    df_clustered = prepare_for_upload(df_clustered)

    df_clustered.to_csv(f'weekly_output.csv', index=False, quoting=csv.QUOTE_ALL)
