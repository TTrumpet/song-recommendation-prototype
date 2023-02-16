# Given that the original Books dataset does has a single CSV file with the ratings,
# we will need to manually split it into train and test sets.

import num_played_to_rating
import preprocess_songcsv
import msdHDF5toCSV
import create_user_data
import pandas as pd
import numpy as np

# Load the Ratings CSV file
df = pd.read_csv('Datasets/Music/train_triplets.txt',
                 delimiter='\t', names=['User-ID', 'SongID', 'Rating'])

# Create Song CSV file
msdHDF5toCSV.main()
print("Done creating SongCSV.csv")

# Load Songs CSV file
songs = pd.read_csv('Datasets/Music/SongCSV.csv')

# preprocess SongCSV to remove trailing characters
preprocess_songcsv.remove_trailing_characters(songs)
print("Done preprocessing SongCSV.csv")

# preprocess 'number of times played' to 'rating'
num_played_to_rating.main()
df = pd.read_csv('Datasets/Music/train_triplets.csv')
print("Done creating train_triplets.csv")

# create user data CSV
create_user_data.main()
print("Done creating user_data.csv")

songs = pd.read_csv('Datasets/Music/SongCSV.csv')
users = pd.read_csv('Datasets/Music/user_data.csv')

# Remove ratings for songs not in songs.csv
df = df[df['SongID'].isin(songs['SongID'])]
# Remove ratings for users not in Users.csv
df = df[df['User-ID'].isin(users['User-ID'])]

# Set random state for reproducibility
random_state = 0

# Split the data into train and test sets
train = df.sample(frac=0.8, random_state=random_state)
test = df.drop(train.index)

# Save the train and test sets to CSV files
train.to_csv('Datasets/Music/Ratings_train.csv', index=None)
test.to_csv('Datasets/Music/Ratings_test.csv', index=None)
