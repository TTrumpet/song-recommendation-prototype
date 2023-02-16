import pandas as pd
import numpy as np


def main():

    # Load the files
    df = pd.read_csv('Datasets/Music/train_triplets.txt',
                     delimiter='\t', names=['User-ID', 'SongID', 'Rating'])
    songs = pd.read_csv('Datasets/Music/SongCSV.csv')
    users = pd.read_csv('Datasets/Music/user_data.csv')

    df_new_ratings = pd.DataFrame(columns=['Rating'])

    for i in range(len(users)):
        id = users["User-ID"][i]

        #df_subset = df[df["User-ID"] == id]
        df_subset = df.loc[df['User-ID'] == id]

        # get z-score
        df_subset = pd.DataFrame((df_subset['Rating'] - df_subset['Rating'].mean()
                                  ) / df_subset['Rating'].std(), columns=['Rating'])

        # fill Nan with 0
        df_subset.fillna(0, inplace=True)

        # cut (clip) greater than 3 or lower than -3
        df_subset['Rating'] = df_subset['Rating'].clip(-3.0, 3.0)

        # convert to range 0-5
        df_subset['Rating'] = df_subset['Rating'].apply(
            lambda x: (5/6) * (x+3))

        df_new_ratings = pd.concat([df_new_ratings, df_subset])
        print(df_new_ratings)

    df['Ratings'] = df_new_ratings['Rating']
    print(df)
    # df.to_csv('Datasets/Music/train_triplets.csv')
    output_file = open('Datasets/Music/train_triplets.txt', 'w')
    output_file.write(df)
    output_file.close()
