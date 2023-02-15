import pandas as pd
import numpy as np


# Load the files
df = pd.read_csv('Datasets/Music/train_triplets.txt',
                 delimiter='\t', names=['User-ID', 'SongID', 'Rating'])
songs = pd.read_csv('Datasets/Music/SongCSV.csv')
users = pd.read_csv('Datasets/Music/user_data.csv')

df_new_ratings = pd.DataFrame(columns=['Rating'])

# def main():
#

for i in range(len(users)):
    id = users["User-ID"][i]
    #df_subset = df[df["User-ID"] == id]
    df_subset = df.loc[df['User-ID'] == id]

    # get z-score
    df_subset = pd.DataFrame((df_subset['Rating'] - df_subset['Rating'].mean()
                              ) / df_subset['Rating'].std(), columns=['Rating'])

    # fills any missing values with 0
    # happens in cases where all plays are the same (no normal distribution)
    # or there is only one row, the user has only listened to one song
    df_subset.fillna('0', inplace=True)

    df_new_ratings = pd.concat([df_new_ratings, df_subset])
    print(df_new_ratings)

print(df_new_ratings)
df['Ratings'] = df_new_ratings['Rating']
print(df)
