import pandas as pd
import numpy as np


def main():

    # Load the files
    df = pd.read_csv('Datasets/Music/train_triplets.txt',
                     delimiter='\t', names=['User-ID', 'SongID', 'Rating'])

    users = pd.read_csv('Datasets/Music/user_data.csv')

    df_new_ratings = pd.DataFrame(columns=['Rating'])

    for i in range(len(users)):

        df_subset = df.loc[lambda df: df['User-ID'] == users['User-ID'][i]]

        # get z-score
        df_subset_ratings = pd.DataFrame((df_subset['Rating'] - df_subset['Rating'].mean()
                                          ) / df_subset['Rating'].std(), columns=['Rating'])

        # fill Nan with 0
        #df_subset.fillna(0, inplace=True)

        # cut (clip) greater than 3 or lower than -3
        #df_subset['Rating'] = df_subset['Rating'].clip(-3.0, 3.0)

        # convert to range 0-5
        # df_subset['Rating'] = df_subset['Rating'].apply(
        #    lambda x: (5/6) * (x+3))

        df_new_ratings = pd.concat([df_new_ratings, df_subset_ratings])
        print(df_new_ratings)

    df['Ratings'] = df_new_ratings['Rating']

    # fill NaN with 0
    df.fillna(0, inplace=True)

    # cut range to between -3 and 3
    df['Rating'] = df['Rating'].clip(-3.0, 3.0)

    # change range to between 0 and 5
    df['Rating'] = df['Rating'].apply(lambda x: (5/6) * (x+3))

    print(df)
    df.to_csv('Datasets/Music/train_triplets.csv')
    # output_file = open('Datasets/Music/train_triplets.txt', 'w')
    # output_file.write(df)
    # output_file.close()


main()
