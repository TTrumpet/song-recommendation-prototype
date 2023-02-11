import pandas as pd
import numpy as np


def remove_trailing_characters(songs):
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
