import pandas as pd
import numpy as np


# Load the files
df = pd.read_csv('Datasets/Music/train_triplets.txt',
                 delimiter='\t', names=['User-ID', 'SongID', 'Rating'])
songs = pd.read_csv('Datasets/Music/SongCSV.csv')
users = pd.read_csv('Datasets/Music/user_data.csv')

# == users.loc['User-ID'][i]]['song-id']['rating'])
