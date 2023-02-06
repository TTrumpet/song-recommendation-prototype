import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_songs, embedding_size=32):
        super(MatrixFactorization, self).__init__()
        self.num_users = num_users
        self.num_songs = num_songs
        self.embedding_size = embedding_size

        # Create the embedding layers for our users and books
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.song_embedding = nn.Embedding(num_songs, embedding_size)

        # Initialize the embeddings with a normal distribution
        self.user_embedding.weight.data.normal_(0, 0.1)
        self.song_embedding.weight.data.normal_(0, 0.1)

    def forward(self, user_id, song_id):
        # Get the embeddings for the user and book
        user_embedding = self.user_embedding(user_id)
        song_embedding = self.song_embedding(song_id)

        # Compute the dot product between the embeddings
        dot_product = torch.sum(user_embedding * song_embedding, dim=1)

        # Pass the dot product through a sigmoid function
        output = torch.sigmoid(dot_product)

        # Scale the output to be between 0 and 10
        rating = 10 * output

        return rating


# ------------------------------- Random Models ------------------------------ #
# Let's create a few random models to get the idea of how to create a model in PyTorch
# by extending the nn.Module class.

# Always 5/10 stars
class Always5(nn.Module):
    def __init__(self, num_users, num_songs, embedding_size=32):
        super(Always5, self).__init__()
        self.num_users = num_users
        self.num_songs = num_songs

    def forward(self, user_id, song_id):
        # Always return 5 stars
        return torch.tensor(5.0, dtype=torch.float32)

# Randomly choose a rating between 0 and 10


class RandomStars(nn.Module):
    def __init__(self, num_users, num_songs, embedding_size=32):
        super(RandomStars, self).__init__()
        self.num_users = num_users
        self.num_songs = num_songs

    def forward(self, user_id, song_id):
        # Return a random rating between 0 and 10
        return torch.tensor(torch.randint(0, 10, (1,)), dtype=torch.float32)

# Randomly choose a rating between 0 and 10 BUT... pass it through a Neural Network B)
# (shouldn't make a difference, but hey, why not?)


class RandomStarsNN(nn.Module):
    def __init__(self, num_users, num_songs, embedding_size=32):
        super(RandomStarsNN, self).__init__()
        self.num_users = num_users
        self.num_songs = num_songs
        self.fc1 = nn.Linear(1, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, user_id, song_id):
        # Return a random rating between 0 and 10
        x = torch.tensor(torch.randint(0, 10, (1,)), dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return 10 * x


if __name__ == "__main__":
    # ---------------------- Quick test of our model --------------------- #
    # Create a model
    model = MatrixFactorization(
        num_users=100, num_songs=100, embedding_size=32)

    # Create some dummy data
    user_id = torch.LongTensor([1, 2, 3, 4, 5])
    song_id = torch.LongTensor([27, 32, 45, 12, 1])

    # Get the output from our models
    output = model(user_id, song_id)
