import pandas as pd
import numpy as np


def main():

    # Load the files
    df_0 = pd.read_csv('Datasets/Music/train_triplets.txt',
                       delimiter='\t', names=['User-ID', 'SongID', 'Rating'])

    df = df_0

    df = df.drop(['SongID'], axis=1)

    # group by same user id
    group = df.groupby('User-ID')

    # get mean and standard deviation
    group = pd.DataFrame(group['Rating'].agg([np.mean, np.std]))

    df = pd.merge(df, group, on='User-ID', how='outer')

    # get z-score
    df['Rating'] = (df['Rating'] - df['mean']) / df['std']

    # fill NaN with 0
    df['Rating'].fillna(0, inplace=True)

    # cut range to between -3 and 3
    df['Rating'] = df['Rating'].clip(-3.0, 3.0)

    # change range to between 0 and 5
    df['Rating'] = df['Rating'].apply(lambda x: (5/6) * (x+3))

    df_0['Rating'] = df['Rating']

    df.to_csv('Datasets/Music/train_triplets.csv')
