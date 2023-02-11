import pandas as pd
import numpy as np


# Load the Ratings CSV file
df = pd.read_csv('Datasets/Music/train_triplets.txt', delimiter='\t')
df.to_csv('Datasets/Music/train_triplets.csv', index=None)
print("train_triplets")
print(df)

# Remove ratings whose User ID or Book ID is not in the Users or Books CSV file
songs = pd.read_csv('Datasets/Music/SongCSV.csv')

userData = pd.read_csv('Datasets/Music/Users.csv', usecols=['User-ID'])
userData.to_csv('Datasets/Music/user_data.csv')
users = pd.read_csv('Datasets/Music/user_data.csv')
print("User IDs")
print(users)

for i in range(len(users['User-ID'])):
    df['Rating'] = df['User-ID'].iloc[i]

# == users.loc['User-ID'][i]]['song-id']['rating'])
