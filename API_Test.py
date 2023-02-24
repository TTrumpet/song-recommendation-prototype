from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_songs, embedding_size=32):
        super(MatrixFactorization, self).__init__()
        self.num_users = num_users
        self.num_songs = num_songs
        self.embedding_size = embedding_size

        # Create the embedding layers for our users and songs
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.song_embedding = nn.Embedding(num_songs, embedding_size)

        # Initialize the embeddings with a normal distribution
        self.user_embedding.weight.data.normal_(0, 0.1)
        self.song_embedding.weight.data.normal_(0, 0.1)

    def forward(self, user_id, song_id):
        # Get the embeddings for the user and song
        user_embedding = self.user_embedding(user_id)
        song_embedding = self.song_embedding(song_id)

        # Compute the dot product between the embeddings
        dot_product = torch.sum(user_embedding * song_embedding, dim=1)

        # Pass the dot product through a sigmoid function
        output = torch.sigmoid(dot_product)

        # Scale the output to be between 0 and 10
        rating = 10 * output

        return rating

    def predict_ratings(self, user_embedding, song_ids):
        # Get the embeddings for the songs
        song_embeddings = self.song_embedding(song_ids)

        # Compute the dot product between the embeddings
        dot_product = torch.sum(user_embedding * song_embeddings, dim=1)

        # Pass the dot product through a sigmoid function
        output = torch.sigmoid(dot_product)

        # Scale the output to be between 0 and 10
        ratings = 10 * output

        return ratings


def getIDLabelEncoder(path):
    """Given a path to a JSON file containing an sklearn LabelEncoder,
    load the LabelEncoder and return it. This LabelEncoder will be used to
    encode the Media IDs.

    Args:
        path (str): Path to the JSON file containing the LabelEncoder data.

    Returns:
        sklearn.preprocessing.LabelEncoder: LabelEncoder for the Media IDs.
    """
    with open(path, 'r') as f:
        id_encoder = LabelEncoder().fit(json.load(f))

    return id_encoder


def get_n_random_ids(media_csv_path, n=10):
    """Given a CSV file containing the media(names of the books/movies/songs/videogames/etc and their IDs),
    return an list of n random IDs.

    Args:
        media_csv_path (str): Path to the CSV file containing the media.
        n (int, optional): Number of random IDs to return. Defaults to 10.

    Returns:
        list: List of n random IDs.
    """
    # Load the CSV file
    media = pd.read_csv(media_csv_path)

    # Get the IDs
    ids = media['ID'].values

    # Get n random IDs
    random_ids = np.random.choice(ids, size=n, replace=False)

    return random_ids


def get_random_ratings(ids):
    """Given a list of Media IDs, return a list of random ratings for each ID
    in the form of a list of tuples (ID, rating).

    Args:
        ids (list): List of IDs.

    Returns:
        list: List of tuples (ID, rating).
    """
    # Get random ratings (0 - 10)
    ratings = np.random.randint(0, 11, size=len(ids))

    # Create a list of tuples (ID, rating)
    random_ratings = list(zip(ids, ratings))

    return random_ratings


def getNearestNeighborEmbedding(model, media_ratings):
    """Given a trained model and a list of Media IDs and their ratings for a user,
    look through your dataset and find the user that has the most similar ratings
    to the user you are trying to predict for. Then, return the embedding for that user.

    Args:
        model (MatrixFactorization): Trained model.
        media_ratings (list): List of tuples (ID, rating).

    Returns:
        torch.Tensor (embedding_size): Embedding for the user that has the most similar
                                       ratings to the user you are trying to predict for.
    """

    # run media_ratings through model for new_user
    new_user = model(media_ratings)

    # transform new_user into embedding (?)

    # store trained embeddings
    embeddings = nn.Embedding.from_pretrained(model.state_dict())

    # find cosine similarity between new user and all other user embeddings
    # batch cosine similarity?
    similarity_scores = pd.DataFrame(
        cosine_similarity(new_user, embeddings), columns=['Score'])

    # return most similar user's embeddings
    return similarity_scores.sort_values('Score', ascending=False)[0]


def getApproximateEmbedding(model, media_ratings):
    """Given a trained model and a list of Media IDs and their ratings for a user,
    compute the approximate embedding for that user. You can do this by getting the
    embeddings for each media that the user has rated minimising the loss between
    the predicted rating and the actual rating using the predict_ratings function.

    Try starting with a random embedding for the user and then iteratively updating
    the embedding to minimise the loss. Feel free to experiment with other methods!
    At the end of the day, the goal is to get a good embedding for the user.

    Args:
        model (MatrixFactorization): Trained model.
        media_ratings (list): List of tuples (ID, rating).

    Returns:
        torch.Tensor (embedding_size): Approximate embedding for the user.
    """

    ## Implement this function too :) ##
    pass


def test_model(model_path, media_csv_path, label_encoder_path, number_users=100, number_predictions=10):
    """ Tests your trained model by generating random ratings for a random number of users
    and then trying to make predictions for those users.

    Args:
        model_path (str): Path to the trained model.
        media_csv_path (str): Path to the CSV file containing the media IDs and names.
        label_encoder_path (str): Path to the JSON file containing the LabelEncoder data.
        number_users (int, optional): Number of random users to generate ratings for. Defaults to 100.
        number_predictions (int, optional): Number of medias to make predictions for. Defaults to 10.
    """

    # Load the model
    # model = MatrixFactorization(...)

    # Load the model's state dictionary (trained weights)
    model.load_state_dict(torch.load(model_path))

    # Load the ID LabelEncoder
    id_encoder = getIDLabelEncoder(label_encoder_path)

    # Set the model to evaluation mode
    model.eval()

    time_start = time.time()

    # Loop through the number of users
    for i in range(number_users):
        # Generate random IDs for the Media
        random_ids = get_n_random_ids(media_csv_path, n=number_predictions)

        # Encode the IDs
        ids_encoded = id_encoder.transform(random_ids)

        # Generate random ratings
        random_ratings = get_random_ratings(ids_encoded)

        # Generate random IDs for medias that the user has not rated
        # but we want to make predictions for
        ids_to_predict = get_n_random_ids(media_csv_path, n=number_predictions)

        # Get the nearest neighbor embedding
        nearest_neighbor_embedding = getNearestNeighborEmbedding(
            model, random_ratings)

        # Get the approximate embedding
        approximate_embedding = getApproximateEmbedding(model, random_ratings)

        # Make predictions for the user
        predictions_nn = model.predict_ratings(
            nearest_neighbor_embedding, ids_to_predict)
        predictions_aprox = model.predict_ratings(
            approximate_embedding, ids_to_predict)

        # Print the predictions
        print(f"==================== User {i} ====================")
        print(f"Nearest Neighbor Predictions: {predictions_nn}")
        print(f"Approximate Predictions: {predictions_aprox}")

    time_end = time.time()

    print(f"Time taken: {time_end - time_start} seconds")


if __name__ == "__main__":
    # Path to the trained model
    model_path = "path/to/your/model.pt"

    # Path to the CSV file containing the media IDs and names
    media_csv_path = "path/to/your/media.csv"

    # Path to the JSON file containing the LabelEncoder data
    label_encoder_path = "path/to/your/label_encoder.json"

    # Number of random users to generate ratings for
    number_users = 100

    # Number of medias to make predictions for
    number_predictions = 50

    test_model(model_path, media_csv_path, label_encoder_path,
               number_users, number_predictions)
