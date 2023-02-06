# Given that the original Books dataset does has a single CSV file with the ratings,
# we will need to manually split it into train and test sets.

import num_played_to_rating
import pandas as pd
import numpy as np

# Load the Ratings CSV file
df = pd.read_csv('Datasets/Music/train_triplets.csv')
print(df)

# Remove ratings whose User ID or Book ID is not in the Users or Books CSV file
songs = pd.read_csv('Datasets/Music/SongCSV(2).csv')

users = pd.read_csv('Datasets/Music/user_data.csv')
print(users)

# preprocess 'number of times played' to 'rating'


# Remove ratings for books not in Books.csv
df = df[df['song_id'].isin(songs['SongID'])]
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
