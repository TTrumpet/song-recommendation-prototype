import pandas as pd
import numpy as np

# Load the files
df = pd.read_csv('Datasets/Music/train_triplets.txt',
                 delimiter='\t', names=['User-ID', 'SongID', 'Rating'])
songs = pd.read_csv('Datasets/Music/SongCSV.csv')


def create_user_data():
    # Load unique users
    users = df['User-ID'].unique()
    users = pd.DataFrame(users, columns=['User-ID'])
    users.to_csv("Datasets/Music/user_data.csv")


def remove_trailing_characters():
    # for columns SongID, AlbumName, ArtistID, ArtistLocation, ArtistName, Title
    songs['SongID'] = songs['SongID'].str.replace('b\'', "")
    songs['SongID'] = songs['SongID'].str.replace('\'', "")

    songs['AlbumName'] = songs['AlbumName'].str.replace('b\'', "")
    songs['AlbumName'] = songs['AlbumName'].str.replace('\'', "")

    songs['ArtistID'] = songs['ArtistID'].str.replace('b\'', "")
    songs['ArtistID'] = songs['ArtistID'].str.replace('\'', "")

    songs['ArtistLocation'] = songs['ArtistLocation'].str.replace('b\'', "")
    songs['ArtistLocation'] = songs['ArtistLocation'].str.replace('\'', "")

    songs['ArtistName'] = songs['ArtistName'].str.replace('b\'', "")
    songs['ArtistName'] = songs['ArtistName'].str.replace('\'', "")

    songs['Title'] = songs['Title'].str.replace('b\'', "")
    songs['Title'] = songs['Title'].str.replace('\'', "")

    songs.to_csv('Datasets/Music/SongCSV.csv')
