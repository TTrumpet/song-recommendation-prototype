import argparse
import os
from os.path import exists
import csv
import json
import torch

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from config import getopt

import glob
import random

from collections import Counter


class Music(Dataset):
    def __init__(self, split='train', opt=None):
        # Set random seed for reproducibility
        np.random.seed(0)

        # Set the split (train or test)
        self.split = split

        # Load the Ratings CSV file (User-ID | ISBN | Book-Rating)
        if split == 'train':
            self.ratings = pd.read_csv('Datasets/Music/Ratings_train.csv')
        if split == 'test':
            self.ratings = pd.read_csv('Datasets/Music/Ratings_test.csv')

        # Get data of all users and books
        self.all_users = pd.read_csv(
            'Datasets/Music/user_data.csv', dtype={'User-ID': str})

        self.all_songs = pd.read_csv('Datasets/Music/SongCSV.csv', dtype={'SongNumber': int, 'SongID': str, 'AlbumID': str, 'AlbumName': str, 'ArtistID': str, 'ArtistLatitude': float, 'ArtistLocation': str,
                                                                          'ArtistLongitude': float, 'ArtistName': str, 'Danceability': float, 'Duration': float, 'KeySignature': int, 'KeySignatureConfidence': float, 'Tempo': float, 'TimeSignature': int, 'TimeSignatureConfidence': float, 'Title': str, 'Year': int})

        # Note: Including the dtype parameter in the pd.read_csv function (^) is not necessary,
        # but it is good practice to do so, as it will prevent pandas from having to infer
        # the data type of each column, which can be slow for large datasets.

        # Get (User ID, Song ID, User Rating) tuples
        self.user_ids = self.ratings['User-ID'].values
        self.song_ids = self.ratings['SongID'].values
        self.ratings = self.ratings['Rating'].values

        # Set general attributes
        self.num_users = len(self.all_users)
        self.num_songs = len(self.all_songs)

        # Since we our ids are random values, we need to map them to a range of
        # integers starting from 0 that we can later use to index into our embedding
        # in our matrix factorization model. We will use the LabelEncoder class from
        # scikit-learn to do this for us.
        self.user_id_encoder = LabelEncoder().fit(
            self.all_users['User-ID'].values)
        self.song_id_encoder = LabelEncoder().fit(
            self.all_songs['SongID'].values)

        # Save trained label encoders as JSON files

        self.index_user_ids = self.user_id_encoder.transform(self.user_ids)
        self.index_song_ids = self.song_id_encoder.transform(self.song_ids)

        num_users = len(self.user_id_encoder.classes_)
        user_ids = self.user_id_encoder.classes_.tolist()

        num_songs = len(self.song_id_encoder.classes_)
        song_ids = self.song_id_encoder.classes_.tolist()

        with open('label_encoder.json', 'w') as f:
            json.dump({'num_users': num_users, 'user_ids': user_ids, 'num_songs': num_songs, 'song_ids': song_ids}, f)
        
        self.index_user_ids = self.user_id_encoder.transform(self.user_ids)
        self.index_song_ids = self.song_id_encoder.transform(self.song_ids)
        
        print("Loaded data, total number of ratings: ", len(self.ratings))
    def __getitem__(self, idx):
        # The __getitem__ method is used to get a single item from the dataset
        # given an index. It is used by the DataLoader class to create batches of data.
        # Let's think, what do we need to return for each item?

        # Given this arbitrary index we will return the user ID of the user who rated
        # the book, the book ID of the book that was rated, and the rating that the
        # user gave to the book. We will also return the encoded user ID and book ID
        # as well, which we will use to index into our embedding matrix in our model.

        # Get the user ID, book ID, and rating
        user_id = self.user_ids[idx]
        song_id = self.song_ids[idx]
        rating = self.ratings[idx]

        # Convert Rating to Torch Tensor (fancy array)
        rating = torch.tensor(rating, dtype=torch.float32)

        # Encode the user ID and song ID
        index_user_id = self.index_user_ids[idx]
        index_song_id = self.index_song_ids[idx]

        return index_user_id, index_song_id, rating, user_id, song_id

    def __len__(self):
        # The __len__ method is used to get the total number of items in the dataset.
        # (this is how the dataloader knows how many batches to create, and which indices
        # are legal to use when calling __getitem__!)
        return len(self.ratings)


if __name__ == "__main__":

    # ---------------------- Testing our Songs Dataset class --------------------- #
    parser = argparse.ArgumentParser()
    opt = getopt()  # Get the options from the config file

    # Create the dataset object and pass it to the DataLoader class
    dataset = Music(opt=opt)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=5, shuffle=True, drop_last=False)

    # Iterate through the dataloader to get the first batch of data
    for i, (encoded_user_id, encoded_song_id, rating, user_id, song_id) in enumerate(dataloader):
        print("Batch: ", i)
        print("Encoded User ID: ", encoded_user_id)
        print("Encoded Song ID: ", encoded_song_id)
        print("Rating: ", rating)
        print("User ID: ", user_id)
        print("Song ID: ", song_id)

        break  # Break the loop after the first batch
